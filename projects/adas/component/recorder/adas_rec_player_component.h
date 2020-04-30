#ifndef ADAS_REC_PLAYER_COMPONENT_H_
#define ADAS_REC_PLAYER_COMPONENT_H_

#include <memory>

#include "cyber/cyber.h"
#include "cyber/timer/timer.h"
#include "cyber/common/time_conversion.h"
#include "cyber/class_loader/class_loader.h"
#include "cyber/component/component.h"

#include "cyber/parameter/parameter_client.h"
#include "cyber/record/record_writer.h"
#include "cyber/proto/topology_change.pb.h"
#include "cyber/proto/record.pb.h"
#include "cyber/message/raw_message.h"
#include "cyber/record/file/record_file_reader.h"


#include "projects/adas/proto/adas_record.pb.h"
#include "projects/adas/component/recorder/player/player.h"
#include "projects/adas/proto/adas_config.pb.h"
#include "projects/adas/proto/adas_record.pb.h"
#include "projects/adas/configs/config_gflags.h"

namespace watrix
{
namespace projects
{
namespace adas
{

using namespace apollo::cyber;
using namespace watrix::projects::adas::proto;

using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using apollo::cyber::Node;
using apollo::cyber::ParameterClient;
using apollo::cyber::ReaderBase;
using apollo::cyber::base::Connection;
using apollo::cyber::message::RawMessage;
using apollo::cyber::proto::ChangeMsg;
using apollo::cyber::proto::RoleAttributes;
using apollo::cyber::proto::RoleType;
using apollo::cyber::service_discovery::ChannelManager;
using apollo::cyber::service_discovery::TopologyManager;
using apollo::cyber::Timer;
using apollo::cyber::TimerOption;

using apollo::cyber::record::HeaderBuilder;
using apollo::cyber::record::RecordWriter;
using apollo::cyber::record::PlayParam;
using apollo::cyber::record::RecordFileReader;
using apollo::cyber::record::Player;
using apollo::cyber::record::Header;

/**
 * @brief  回放播放模块
 * 
 */
class AdasRecPlayerComponent : public Component<>
{

public:
    AdasRecPlayerComponent();
    ~AdasRecPlayerComponent();
    AdasRecPlayerComponent(const AdasRecPlayerComponent &) =
        delete;
    AdasRecPlayerComponent &operator=(
        const AdasRecPlayerComponent &) = delete;
    bool Init() override;
    bool Stop();


private:
    bool InitConfig();
    void FileQueryCallback(const std::shared_ptr<FilesQueryParam> &query,
                           const std::shared_ptr<FilesAnswerParam> &answer);

    void RecordQueryCallback(const std::shared_ptr<RecordQueryParam> &query,
                             const std::shared_ptr<RecordAnswerParam> &answer);
    void EnumRecordFiles();


    watrix::projects::adas::proto::RecPlayerConfig config_;
    std::mutex enum_file_mutex_;
    std::shared_ptr<Service<FilesQueryParam, FilesAnswerParam>> file_q_service_ = nullptr;
    std::shared_ptr<Service<RecordQueryParam, RecordAnswerParam>> record_q_service_ = nullptr;
    apollo::cyber::Timer file_enum_timer_;
    //record存档文件
    std::vector<std::string> records_files_;

    FilesAnswerParam record_file_info_;

    std::string record_save_dir_;

    std::string record_q_servicename_;
    std::string file_q_servicename_;
};

CYBER_REGISTER_COMPONENT(AdasRecPlayerComponent)

} // namespace adas
} // namespace projects
} // namespace watrix

#endif