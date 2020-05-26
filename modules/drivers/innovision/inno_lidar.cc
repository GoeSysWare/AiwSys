/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include <cmath>
#include <ctime>
#include <string>

#include "cyber/cyber.h"

#include "modules/drivers/innovision/inno_lidar.h"
#include "modules/drivers/innovision/proto/config.pb.h"
#include "modules/drivers/innovision/proto/inno_lidar.pb.h"

namespace apollo
{
    namespace drivers
    {
        namespace innovision
        {

            // uint64_t InnoLidarDriver::sync_counter = 0;
            //报警回调
            void inno_lidar_alarm_callback(int lidar_handle, void *context,
                                           inno_alarm error_level, inno_alarm_code error_code,
                                           const char *error_message)
            {
                InnoLidarDriver *driver = static_cast<InnoLidarDriver *>(context);
                if (error_level >= INNO_ALARM_CRITICAL)
                {
                    AERROR << "Innovision Lidar Disconnected";
                    driver->is_alive_ = false;
                }
                //暂时啥也不干
                return;
            }
            double inno_lidar_host_time_in_second_t(void *context)
            {

                return apollo::cyber::Time::Now().ToSecond();
            }
            //接收Lidar数据的回调
            int inno_lidar_frame_callback(int lidar_handle, void *context,
                                          struct inno_frame *frame)
            {
                //让实际回调干活,由于frame指针是底层设备驱动释放，
                //而执行过程为了减少内存拷贝，没有采用深拷贝，所以这里不能用异步模式
                // 否则frame指针的生命周期可能在异步执行前就结束了
                //这里是一个同步阻塞过程，slot函数不能太耗时，否则把回调阻塞过久

                InnoLidarDriver *driver = static_cast<InnoLidarDriver *>(context);
                driver->frame_signal_(frame);
                return 0;
            }

            InnoLidarDriver::InnoLidarDriver(const proto::Config &config) : config_(config)
            {
                is_alive_ = false;
                keep_alive_ = true;
                is_start_ = false;
            }
            InnoLidarDriver::~InnoLidarDriver()
            {
                if (reconnect_thread_.joinable())
                {
                    reconnect_thread_.join();
                }

                Stop();
                inno_lidar_close(inno_handle_);
                frame_signal_.DisconnectAllSlots();
            }

            bool InnoLidarDriver::Init()
            {
                //频率参数预留
                double frequency = (config_.rpm() / 60.0); // expected Hz rate

                config_.set_npackets(static_cast<int>(ceil(packet_rate_ / frequency)));
                AINFO << "publishing " << config_.npackets() << " packets per scan";

                //连接设备
                if (!InitDevice())
                    return false;

                //创建事件回调
                inno_lidar_set_callbacks(inno_handle_, inno_lidar_alarm_callback,
                                         inno_lidar_frame_callback, inno_lidar_host_time_in_second_t, this);

                return true;
            }
            bool InnoLidarDriver::InitDevice()
            {
                std::string frame_id = config_.frame_id();
                std::string ip = config_.firing_data_ip();
                int port = config_.firing_data_port();
                //保证不超过32个字符
                frame_id = frame_id.substr(0, 31);
                //创建lidar链接
                inno_handle_ = inno_lidar_open_live(frame_id.c_str(), ip.c_str(), port, true);
                if (inno_handle_ < 0)
                {
                    AERROR << "InnoLidarDriver Init Device Failed :" << frame_id
                           << " ip:" << ip
                           << " port:" << port;
                    return false;
                }
                AINFO << "InnoLidarDriver Init Device:" << frame_id
                      << " ip:" << ip
                      << " port:" << port;
                //设定Lidar标定文件
                if (config_.calibration_online())
                {
                    // 标定文件必须用绝对地址
                    int params = inno_lidar_set_parameters(inno_handle_, inno_mode_.c_str(), "", config_.calibration_file().c_str());
                    if (params != 0)
                    {
                        AERROR << "InnoLidarDriver set_parameters failed:" << config_.calibration_file();
                        return false;
                    }
                }


                reconnect_thread_ = std::thread(&InnoLidarDriver::ReconnectThread, this);

                return true;
            }
            bool InnoLidarDriver::Start()
            {
                //未初始化
                if (inno_handle_ < 0 )
                    return false;
                //已经启动过
                if(is_start_)
                    return true;

                return inno_lidar_start(inno_handle_) == 0 ? is_alive_ = true, is_start_ = true, true : is_start_ = false, false;
            }
            bool InnoLidarDriver::Stop()
            {
                if (inno_handle_ < 0)
                    return false;
                if(is_start_)
                {
                    inno_lidar_stop(inno_handle_);
                    is_start_ = false;
                }
                return true;
            }

            void InnoLidarDriver::SetFiringModel(proto::Mode mode)
            {
                //后续可以在此扩展
                switch (mode)
                {
                case proto::REV_E:
                {
                    inno_mode_ = "REV_E";
                    break;
                }
                case proto::E:
                {
                    inno_mode_ = "E";
                    break;
                }
                default:
                    AERROR << "invalid mode, must be REV_E | E";
                    break;
                }
            }
            //利用signal-slot机制，可以支持一个信号，顺序触发多个回调，
            // 同时也可以解耦，低耦合
            // 本项目目前只用了一个slot
            void InnoLidarDriver::SetConnection(std::function<void(struct inno_frame *)> frame_parse_func)
            {
                frame_signal_.Connect(frame_parse_func);
            }
            //重连线程
            void InnoLidarDriver::ReconnectThread()
            {
                while (!cyber::IsShutdown())
                {
                    if (!is_alive_.load() && inno_handle_ > 0)
                    {
                        //先停止
                        Stop();
                        //如果保持重连
                        if (keep_alive_)
                        {
                            Start();
                        }
                    }
                    cyber::SleepFor(std::chrono::microseconds(5000));
                }
            }

            InnoLidarDriver *InnovisionDriverFactory::CreateDriver(const proto::Config &config)
            {
                InnoLidarDriver *driver = nullptr;
                //目前只支持一款类型，后续可以在此扩展
                switch (config.model())
                {
                case proto::INV300:
                {
                    driver = new InnoLidarDriver(config);
                    driver->SetPacketRate(PACKET_RATE_INV300);
                    driver->SetFiringModel(config.mode());
                    break;
                }
                default:
                    AERROR << "invalid model, must be INV300";
                    break;
                }

                return driver;
            }

        } // namespace innovision
    }     // namespace drivers
} // namespace apollo
