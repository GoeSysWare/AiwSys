

#pragma once

#include <memory>
#include <string>
#include "cyber/base/signal.h"
#include "modules/drivers/innovision/proto/config.pb.h"
#include "modules/drivers/innovision/proto/inno_lidar.pb.h"
#include "inno_lidar_api.h"

using  apollo::cyber::base::Signal;

namespace apollo
{
    namespace drivers
    {
        namespace innovision
        {

            constexpr double PACKET_RATE_INV300 = 754;


            /**
             * @brief  Inno Lidar驱动
             * 
             *  Innovison 公司的Lidar驱动
             */
            class InnoLidarDriver
            {
            public:
                explicit InnoLidarDriver(const proto::Config &config);
                virtual ~InnoLidarDriver();

                virtual bool Init();
                virtual bool Start();
                virtual bool Stop();
                void SetPacketRate(const double packet_rate) { packet_rate_ = packet_rate; }
                void SetFiringModel( proto::Mode mode);
                void SetConnection(std::function<void(struct inno_frame *) > frame_parse_func);
            protected:
                proto::Config config_;
                std::string topic_;
                double packet_rate_ = 0.0;
                //Lidar句柄
                int inno_handle_ = -1;
                //Lidar是否在线
                std::atomic_bool is_alive_;
                //Lidar是否已经启动
                std::atomic_bool is_start_;

                std::string inno_mode_;
                //是否保持断线重连
                bool keep_alive_;

              std::thread reconnect_thread_;
    
                friend void inno_lidar_alarm_callback(int lidar_handle, void *context,
                                               enum inno_alarm error_level, enum inno_alarm_code error_code,
                                               const char *error_message);
                friend int inno_lidar_frame_callback(int lidar_handle, void *context,
                                              struct inno_frame *frame);

               virtual bool InitDevice();
            private:
               Signal<struct inno_frame *> frame_signal_;
                 void ReconnectThread();
            };

            //这是给未来型号预留的代码，新型号的模板扩展
            class InnoLidarDriver_XXX : public InnoLidarDriver
            {
            public:
                explicit InnoLidarDriver_XXX(const proto::Config &config) : InnoLidarDriver(config) {}
                ~InnoLidarDriver_XXX() {}

                bool Init() override {return false;}
                bool Start() override {return false;}
            };

            //根据配置信息得到相对应的型号驱动
            class InnovisionDriverFactory
            {
            public:
                static InnoLidarDriver *CreateDriver(const proto::Config &config);
            };

        } // namespace innovision
    }     // namespace drivers
} // namespace apollo
