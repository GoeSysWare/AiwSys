
#include "projects/adas/component/recorder/adas_rec_player_component.h"


#include "boost/filesystem.hpp"
namespace watrix
{
namespace projects
{
namespace adas
{

AdasRecPlayerComponent::AdasRecPlayerComponent()
{
}
AdasRecPlayerComponent::~AdasRecPlayerComponent()
{
}


bool AdasRecPlayerComponent::InitConfig()
{
       bool ret = GetProtoConfig(&config_);
    if (!ret)
    {
        return false;
    }
    record_save_dir_ = config_.records_save_dir();
    file_q_servicename_ = config_.interface_service().records_file_servicename();
    record_q_servicename_ = config_.interface_service().records_record_servicename();

    return true; 
}

bool AdasRecPlayerComponent::Init()
{
    if (!InitConfig())
    {
        AERROR << "AdasRecPlayerComponent Init Failed";
        return false;
    }
    //创建接口服务，响应客户端查询
    file_q_service_ = node_->CreateService<FilesQueryParam, FilesAnswerParam>(
        file_q_servicename_,
        std::bind(&AdasRecPlayerComponent::FileQueryCallback, this, std::placeholders::_1, std::placeholders::_2));
    record_q_service_ = node_->CreateService<RecordQueryParam, RecordAnswerParam>(
        record_q_servicename_,
        std::bind(&AdasRecPlayerComponent::RecordQueryCallback, this, std::placeholders::_1, std::placeholders::_2));

    //启动定时器监测目录下有多少记录文件
    TimerOption opt;
    opt.oneshot = false;
    //10s
    opt.period = 10000;
    opt.callback = std::bind(&AdasRecPlayerComponent::EnumRecordFiles, this);
    file_enum_timer_.SetTimerOption(opt);
    file_enum_timer_.Start();
    return true;
}

bool AdasRecPlayerComponent::Stop()
{
    file_enum_timer_.Stop();
    return true;
}

void AdasRecPlayerComponent::FileQueryCallback(const std::shared_ptr<FilesQueryParam> &query,
                                      const std::shared_ptr<FilesAnswerParam> &answer)
{

    AERROR << "Client: " << query->client_name() << "call interface  FileQueryCallback, cmd type : " << query->cmd_type();

    //这里需要加锁
    std::lock_guard<std::mutex> lock(enum_file_mutex_);
    //需要知道有哪些文件
    if (query->cmd_type() == RecordCmdType::CMD_FILE_QUERY)
    {
        *answer = record_file_info_;
        answer->set_status(true);
    }
    //关闭后台视频回放播放行为，手动关闭
    else if (query->cmd_type() == RecordCmdType::CMD_RECORD_STOP)
    {

    }
}

void AdasRecPlayerComponent::RecordQueryCallback(const std::shared_ptr<RecordQueryParam> &query,
                                        const std::shared_ptr<RecordAnswerParam> &answer)
{
    ADEBUG << "Client:  call interface  RecordQueryCallback, cmd type : " << query->cmd_type();
    //播放特定的视频文件
    if (query->cmd_type() == RecordCmdType::CMD_RECORD_PLAY)
    {
        std::string filename = query->file_name();

        std::shared_ptr<AdasRecPlayerComponent> share_this =
            std::dynamic_pointer_cast<AdasRecPlayerComponent>(shared_from_this());

        auto player_thread =
            std::make_shared<std::thread>([share_this,filename]() {
                PlayParam play_param;
                play_param.is_play_all_channels = true;
                play_param.is_loop_playback = false;
                play_param.play_rate = 1.0;
                play_param.files_to_play.emplace(filename);
                std::shared_ptr<Player> player(new Player(share_this->node_,play_param));
                player->Init();
                player->Start();
            });
        //分离线程，让player_thread 和线程没有关系，让线程自己自然退出
        // 如果不是分离线程，就会同步阻塞接口函数，让客户端半天才能返回
        player_thread->detach();

        answer->set_status(true);
    }
}

//不采用新增遍历的方式是因为当文件数量超过限制，会循环利用旧文件，
//文件的内部信息一直可能会更新
//这样的话，虽然顺序遍历效率低下，但还是顺序遍历方式才行
void AdasRecPlayerComponent::EnumRecordFiles()
{
    using namespace std;

    namespace fs = boost::filesystem;
    //路径名不会太短
    if (record_save_dir_.size() < 2)
    {
        return;
    }

    if (!fs::exists(record_save_dir_))
    {
        AERROR << "records  is not existed : " << record_save_dir_;
        return;
    }

    records_files_.clear();

    // 遍历配置文件所在的文件夹,得到所有的配置文件名.
    if (fs::is_directory(record_save_dir_))
    {
        fs::directory_iterator end_iter;
        for (fs::directory_iterator dir_itr(record_save_dir_); dir_itr != end_iter; ++dir_itr)
        {
            try
            {
                // 判断合法record文件
                if (dir_itr->path().string().find(std::string(".record")) == std::string::npos)
                    continue;
                records_files_.emplace_back(dir_itr->path().string());
            }
            catch (const std::exception &ex)
            {
                string error_msg = dir_itr->path().string();
                error_msg.append(",enum  error：");
                error_msg.append(ex.what());
                AERROR << error_msg;
            }
        }
    }
    else // must be a file
    {
        AERROR << record_save_dir_ << " is not dir ,may be is file";
        return;
    }

    //这里需要加锁
    std::lock_guard<std::mutex> lock(enum_file_mutex_);

    //清空现有的信息
    record_file_info_.Clear();

    // 遍历文件取得文件信息
    for (auto &file : records_files_)
    {
        RecordFileReader file_reader;
        if (!file_reader.Open(file))
        {
            AERROR << "open record file error. file: " << file;
            continue;
        }
        Header hdr = file_reader.GetHeader();
        auto begin_time_s = static_cast<double>(hdr.begin_time()) / 1e9;
        auto end_time_s = static_cast<double>(hdr.end_time()) / 1e9;
        auto duration_s = end_time_s - begin_time_s;
        auto begin_time_str = apollo::cyber::common::UnixSecondsToString(static_cast<int>(begin_time_s));
        auto end_time_str = apollo::cyber::common::UnixSecondsToString(static_cast<int>(end_time_s));
        auto file_size = static_cast<uint64_t>(hdr.size()) / (1024 * 1024); //MB
        auto is_complete = hdr.is_complete();
        auto channel_number = hdr.channel_number();
        auto msg_number = hdr.message_number();

        //赋值
        FileParam *fileparam = record_file_info_.add_files();

        fileparam->set_file_name(file);
        fileparam->set_start_time(begin_time_str);
        fileparam->set_duration(duration_s);
        fileparam->set_is_completed(is_complete);
        fileparam->set_file_size(file_size);
        fileparam->set_channel_number(channel_number);
        fileparam->set_msg_number(msg_number);

        file_reader.Close();
        //为了提高速度，暂时不提供channel信息
        // fileparam->add_channel_name(hdr.);
    }
    return;
}
} // namespace adas
} // namespace projects
} // namespace watrix