
#include "projects/adas/component/recorder/adas_rec_recorder_component.h"


#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "cyber/class_loader/class_loader.h"
#include "cyber/component/component.h"
#include "cyber/common/time_conversion.h"


#include "projects/adas/component/common/util.h"

using namespace watrix::projects::adas::proto;

namespace watrix
{
namespace projects
{
namespace adas
{

AdasRecRecorderComponent::AdasRecRecorderComponent() : all_channels_(false)
{
}
AdasRecRecorderComponent::~AdasRecRecorderComponent()
{
    Stop();
}

bool AdasRecRecorderComponent::InitConfig()
{
    bool ret = GetProtoConfig(&config_);
    if (!ret)
    {
        return false;
    }
    //需要设置ADAS_PATH的环境变量，
    // 如果没有设置环境变量，则以配置的地址必须为绝对地址
    // 如果设置了环境变量，则可以设置相对地址= 环境变量+配置相对地址
    std::string record_dir =  apollo::cyber::common::GetAbsolutePath(GetAdasWorkRoot(),config_.records_save_dir());

    //组成存档文件名 = 路径+ 前缀 + 当前时间+ .record+ 序号
    output_ = record_dir + "/" + config_.records_filename_suffix() + "-" +
              apollo::cyber::common::UnixSecondsToString(time(nullptr), "%Y%m%d%H%M%S") + ".record";

    //没有目录则创建目录
    boost::filesystem::create_directories(record_dir);


    //每次启动时,是否清空原有的历史记录
    if(config_.records_is_clear_earlier()) apollo::cyber::common::RemoveAllFiles(record_dir);

    record_model_ = config_.records_save_model();

    //取得默认的内置参数
  watrix::projects::adas::proto::InterfaceServiceConfig interface_config;
  apollo::cyber::common::GetProtoFromFile(
    apollo::cyber::common::GetAbsolutePath(watrix::projects::adas::GetAdasWorkRoot(), FLAGS_adas_cfg_interface_file),
        &interface_config);
    //此作为参数服务的客户端，名字需要根服务端一致
    parameter_service_name_ = interface_config.records_parameter_servicename();
    record_parameter_name_ = interface_config.record_parameter_name();
    // 记录的通道名
    //一般用默认的通道名称
    boost::algorithm::split(channel_vec_, interface_config.camera_channels(), boost::algorithm::is_any_of(","));

    return true;
}

bool AdasRecRecorderComponent::Init()
{
    if (!InitConfig())
    {
        AERROR << "AdasRecRecorderComponent Init Failed";
        return false;
    }
    writer_.reset(new RecordWriter(HeaderBuilder::GetHeader()));

    //对存档操作初始化一些设置
    int64_t filesize = config_.records_file_max_size();
    if (filesize < 0 || filesize > 2048)
        filesize = record_model_ == RecorderModel::RECORD_INTERVAL ? 512 : 2048;
    //一个文件最大默认最大容量
    writer_->SetSizeOfFileSegmentation(filesize * 1000);
    int64_t interval = config_.records_file_interval();
    if (interval < 3 || interval > 120)
        interval = record_model_ == RecorderModel::RECORD_INTERVAL ? 5 : 120;
    //一个文件最大保存默认时间间隔
    writer_->SetIntervalOfFileSegmentation(interval);
    //最多默认回放文件
    int64_t num = config_.records_file_num();
    if (num < 0 || num > 512)
        num = record_model_ == RecorderModel::RECORD_INTERVAL ? 99 : 200;
    writer_->SetCountOfFileSegmentation(num);

    if (!writer_->Open(output_))
    {
        AERROR << "Datafile open file error.";
        return false;
    }

    param_client_.reset(new ParameterClient(node_, parameter_service_name_));

    if (!InitReadersImpl())
    {
        AERROR << " _init_readers error.";
        return false;
    }
    message_count_ = 0;
    message_time_ = 0;
    is_started_ = true;
    display_thread_ =
        std::make_shared<std::thread>([this]() { this->ShowProgress(); });
    if (display_thread_ == nullptr)
    {
        AERROR << "init display thread error.";
        return false;
    }
    return true;
}

bool AdasRecRecorderComponent::Stop()
{
    if (!is_started_ || is_stopping_)
    {
        return false;
    }
    is_stopping_ = true;
    if (!FreeReadersImpl())
    {
        AERROR << " _free_readers error.";
        return false;
    }
    param_client_.reset();
    writer_->Close();
    node_.reset();
    if (display_thread_ && display_thread_->joinable())
    {
        display_thread_->join();
        display_thread_ = nullptr;
    }
    is_started_ = false;
    is_stopping_ = false;
    return true;
}

void AdasRecRecorderComponent::TopologyCallback(const ChangeMsg &change_message)
{
    ADEBUG << "ChangeMsg in Topology Callback:" << std::endl
           << change_message.ShortDebugString();
    if (change_message.role_type() != apollo::cyber::proto::ROLE_WRITER)
    {
        ADEBUG << "Change message role type is not ROLE_WRITER.";
        return;
    }

    FindNewChannel(change_message.role_attr());
}

void AdasRecRecorderComponent::FindNewChannel(const RoleAttributes &role_attr)
{
    if (!role_attr.has_channel_name() || role_attr.channel_name().empty())
    {
        AWARN << "change message not has a channel name or has an empty one.";
        return;
    }
    if (!role_attr.has_message_type() || role_attr.message_type().empty())
    {
        AWARN << "Change message not has a message type or has an empty one.";
        return;
    }
    if (!role_attr.has_proto_desc() || role_attr.proto_desc().empty())
    {
        AWARN << "Change message not has a proto desc or has an empty one.";
        return;
    }
    if (!all_channels_ &&
        std::find(channel_vec_.begin(), channel_vec_.end(),
                  role_attr.channel_name()) == channel_vec_.end())
    {
        ADEBUG << "New channel was found, but not in record list.";
        return;
    }
    if (channel_reader_map_.find(role_attr.channel_name()) ==
        channel_reader_map_.end())
    {
        if (!writer_->WriteChannel(role_attr.channel_name(),
                                   role_attr.message_type(),
                                   role_attr.proto_desc()))
        {
            AERROR << "write channel fail, channel:" << role_attr.channel_name();
        }
        InitReaderImpl(role_attr.channel_name(), role_attr.message_type());
    }
}

bool AdasRecRecorderComponent::InitReadersImpl()
{
    std::shared_ptr<ChannelManager> channel_manager =
        TopologyManager::Instance()->channel_manager();

    // get historical writers
    std::vector<apollo::cyber::proto::RoleAttributes> role_attr_vec;
    channel_manager->GetWriters(&role_attr_vec);
    for (auto role_attr : role_attr_vec)
    {
        FindNewChannel(role_attr);
    }

    // listen new writers in future
    change_conn_ = channel_manager->AddChangeListener(
        std::bind(&AdasRecRecorderComponent::TopologyCallback, this, std::placeholders::_1));
    if (!change_conn_.IsConnected())
    {
        AERROR << "change connection is not connected";
        return false;
    }
    return true;
}

bool AdasRecRecorderComponent::FreeReadersImpl()
{
    std::shared_ptr<ChannelManager> channel_manager =
        TopologyManager::Instance()->channel_manager();

    channel_manager->RemoveChangeListener(change_conn_);

    return true;
}

bool AdasRecRecorderComponent::InitReaderImpl(const std::string &channel_name,
                                              const std::string &message_type)
{
    try
    {
        std::weak_ptr<AdasRecRecorderComponent> weak_this =
            std::dynamic_pointer_cast<AdasRecRecorderComponent>(shared_from_this());

        std::shared_ptr<ReaderBase> reader = nullptr;
        auto callback = [weak_this, channel_name](
                            const std::shared_ptr<RawMessage> &raw_message) {
            auto share_this = weak_this.lock();
            if (!share_this)
            {
                return;
            }
            share_this->ReaderCallback(raw_message, channel_name);
        };
        ReaderConfig config;
        config.channel_name = channel_name;
        config.pending_queue_size =
            gflags::Int32FromEnv("CYBER_PENDING_QUEUE_SIZE", 50);
        reader = node_->CreateReader<RawMessage>(config, callback);
        if (reader == nullptr)
        {
            AERROR << "Create reader failed.";
            return false;
        }
        channel_reader_map_[channel_name] = reader;
        return true;
    }
    catch (const std::bad_weak_ptr &e)
    {
        AERROR << e.what();
        return false;
    }
}

void AdasRecRecorderComponent::ReaderCallback(const std::shared_ptr<RawMessage> &message,
                                              const std::string &channel_name)
{
    if (!is_started_ || is_stopping_)
    {
        AERROR << "record procedure is not started or stopping.";
        return;
    }

    if (message == nullptr)
    {
        AERROR << "message is nullptr, channel: " << channel_name;
        return;
    }
    apollo::cyber::Parameter parameter;
    switch (record_model_)
    {
    case RecorderModel::RECORD_INTERVAL:
        param_client_->GetParameter(record_parameter_name_, &parameter);
        if (parameter.AsBool())
        {
            message_time_ = Time::Now().ToNanosecond();
            if (!writer_->WriteMessage(channel_name, message, message_time_))
            {
                AERROR << "write data fail, channel: " << channel_name;
                return;
            }
        }
        break;
    case RecorderModel::RECORD_CONTINUOUS:
        message_time_ = Time::Now().ToNanosecond();
        if (!writer_->WriteMessage(channel_name, message, message_time_))
        {
            AERROR << "write data fail, channel: " << channel_name;
            return;
        }
        break;
    default:
        break;
    }

    message_count_++;
}

//间断记录回放模式下，定时长把parameter给设置为false
//这个时长就是回放的长度
void AdasRecRecorderComponent::ShowProgress()
{
    while (is_started_ && !is_stopping_)
    {

        apollo::cyber::Parameter parameter;
        param_client_->GetParameter(record_parameter_name_, &parameter);
        if (parameter.AsBool())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            apollo::cyber::Parameter set_para(record_parameter_name_, false);
            param_client_->SetParameter(set_para);
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

} // namespace adas
} // namespace projects
} // namespace watrix
