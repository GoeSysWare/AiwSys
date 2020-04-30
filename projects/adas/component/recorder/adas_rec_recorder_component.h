#ifndef ADAS_REC_RECORDER_COMPONENT_H_
#define ADAS_REC_RECORDER_COMPONENT_H_

#include <memory>

#include "cyber/cyber.h"
#include "cyber/class_loader/class_loader.h"
#include "cyber/component/component.h"

#include "cyber/parameter/parameter_client.h"
#include "cyber/record/record_writer.h"
#include "cyber/proto/topology_change.pb.h"
#include "cyber/proto/record.pb.h"
#include "cyber/message/raw_message.h"

#include "projects/adas/proto/adas_record.pb.h"
#include "projects/adas/proto/adas_config.pb.h"


#include "projects/adas/configs/config_gflags.h"

namespace watrix
{
namespace projects
{
namespace adas
{

using namespace apollo::cyber;

using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using apollo::cyber::Node;
using apollo::cyber::ReaderBase;
using apollo::cyber::base::Connection;
using apollo::cyber::message::RawMessage;
using apollo::cyber::proto::ChangeMsg;
using apollo::cyber::proto::RoleAttributes;
using apollo::cyber::proto::RoleType;
using apollo::cyber::record::RecordWriter;
using apollo::cyber::service_discovery::ChannelManager;
using apollo::cyber::service_discovery::TopologyManager;
using apollo::cyber::record::HeaderBuilder;
using apollo::cyber::ParameterClient;
/**
 * @brief  回放记录模块
 * 
 */
class AdasRecRecorderComponent : public Component<>
{

    using ParameterClientPtr= std::shared_ptr<apollo::cyber::ParameterClient> ;
public:
    AdasRecRecorderComponent();
    ~AdasRecRecorderComponent();
    AdasRecRecorderComponent(const AdasRecRecorderComponent &) =
        delete;
    AdasRecRecorderComponent &operator=(
        const AdasRecRecorderComponent &) = delete;
    bool Init() override;
    bool Stop();

private:
    watrix::projects::adas::proto::RecRecorderConfig config_;
    watrix::projects::adas::proto::RecorderModel  record_model_;

    ParameterClientPtr param_client_ = nullptr;
    std::string parameter_service_name_;
    bool is_started_ = false;
    bool is_stopping_ = false;
    std::shared_ptr<RecordWriter> writer_ = nullptr;
    std::shared_ptr<std::thread> display_thread_ = nullptr;
    Connection<const ChangeMsg &> change_conn_;
    std::string output_;
    bool all_channels_ = true;
    std::vector<std::string> channel_vec_;
    apollo::cyber::proto::Header header_;
    std::unordered_map<std::string, std::shared_ptr<ReaderBase>>
        channel_reader_map_;
    uint64_t message_count_;
    uint64_t message_time_;

    bool InitConfig();

    bool InitReadersImpl();

    bool FreeReadersImpl();

    bool InitReaderImpl(const std::string &channel_name,
                        const std::string &message_type);

    void TopologyCallback(const ChangeMsg &msg);

    void ReaderCallback(const std::shared_ptr<RawMessage> &message,
                        const std::string &channel_name);

    void FindNewChannel(const RoleAttributes &role_attr);

    void ShowProgress();
};


CYBER_REGISTER_COMPONENT(AdasRecRecorderComponent)

} // namespace adas
} // namespace projects
} // namespace watrix


#endif
