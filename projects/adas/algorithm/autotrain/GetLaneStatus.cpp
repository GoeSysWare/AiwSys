#include "GetLaneStatus.h"

namespace watrix {
	namespace algorithm {
        double LaneStatus::GetFirstDer(std::vector<double>& param_list, double x){
            double y=0.0;
            int num_size = param_list.size()-1;
            for (int idx=num_size; idx>0; idx--){
                double param = param_list[num_size-idx];
                y += idx*param*pow(x, idx-1);
            }
            return y;
        }

        double LaneStatus::GetSecondDer(std::vector<double>& param_list, double x){
            double y=0.0;
            int num_size = param_list.size()-1;
            for (int idx=num_size; idx>1; idx--){
                double param = param_list[num_size-idx];
                y += idx*(idx-1)*param*pow(x, idx-2);
            }
            return y;
        }

        double LaneStatus::GetCurvature(std::vector<double>& param_list, double x){
            double K = fabs(GetSecondDer(param_list, x)) / sqrt( pow(1+pow(GetFirstDer(param_list, x),2), 3) );
            return K;
        }

        double LaneStatus::GetCurvatureR(std::vector<double>& param_list, double x){
            double K = sqrt( pow(1+pow(GetFirstDer(param_list, x),2), 3) ) / fabs(GetSecondDer(param_list, x));
            return K;
        }

        std::vector<double> LaneStatus::polyfit(std::vector<double>& x, std::vector<double>& y, int n)
        {
            std::vector<double> param_list;
            int size = x.size();
            //unkonw parameters count
            int x_num = n + 1;
            //Mat U,Y
            cv::Mat mat_u(size, x_num, CV_64F);
            cv::Mat mat_y(size, 1, CV_64F);

            for (int i = 0; i < mat_u.rows; ++i)
                for (int j = 0; j < mat_u.cols; ++j)
                {
                    mat_u.at<double>(i, j) = pow(x[i], j);
                }
        
            for (int i = 0; i < mat_y.rows; ++i)
            {
                mat_y.at<double>(i, 0) = y[i];
            }
        
            //Get coefficients mat K
            cv::Mat mat_k(x_num, 1, CV_64F);
            mat_k = (mat_u.t()*mat_u).inv()*mat_u.t()*mat_y;
            for (int idx=n; idx>=0; idx--){
                param_list.push_back(mat_k.at<double>(idx, 0));
            }
            return param_list;
        }

        double LaneStatus::polyfit_predict(std::vector<double>& param_list, double x)
        {
            double y = 0;
            int n = param_list.size()-1;
            for (int j = 0; j <= n; ++j)
            {
                y += param_list[j]*pow(x,n-j);
            }
            return y;
        }

        bool LaneStatus::is_lane_straight(std::vector<dpoints_t>& v_src_dist_lane_points){
            bool result = false;
            std::vector<dpoints_t> v_xy;
            for(int lane_id=0; lane_id<v_src_dist_lane_points.size(); lane_id++){
                const dpoints_t& one_lane_points = v_src_dist_lane_points[lane_id];
                dpoints_t t_xy;
                dpoint_t t_x, t_y;
                // change x,y to y,x; t_x from small to large
                // std::cout << "lane_id:" << lane_id << std::endl;
                for(int idx=one_lane_points.size()-1; idx>=0; idx--){
                    t_x.push_back(one_lane_points[idx][1]);
                    t_y.push_back(one_lane_points[idx][0]);
                    // std::cout << one_lane_points[idx][0] << " " << one_lane_points[idx][1] << std::endl;
                 }
                t_xy.push_back(t_x);
                t_xy.push_back(t_y);
                v_xy.push_back(t_xy);
            }
            // std::cout << "step 0 end" << std::endl;
            //step1: cal one order error
            std::vector<double> y_error_list;
            for(int lane_id=0; lane_id<v_xy.size(); lane_id++){
                const dpoints_t& t_xy = v_xy[lane_id];
                dpoint_t t_x = t_xy[0], t_y = t_xy[1];
                std::vector<double> param_list_1 = polyfit(t_x, t_y, 1);
                double y_error = 0.0;
                for (int idx=0; idx<t_x.size(); idx++)
                    y_error += pow((t_y[idx] - polyfit_predict(param_list_1, t_x[idx])), 2);
                y_error_list.push_back(y_error);
            }
            // std::cout << "step 1 end" << std::endl;
            std::cout << "error:" << y_error_list[0] << " " << y_error_list[1] << std::endl;
            if (y_error_list[0] >= 10.0 || y_error_list[1] >= 10.0){
                result = false;
            }
            else{
                result = true;
            }
            
            return result;
        }

        int LaneStatus::GetLaneStatus(
            dpoints_t& v_param_list, std::vector<dpoints_t>& v_src_dist_lane_points,
            dpoints_t& curved_point_list, std::vector<double>& curved_r_list
        ){
            int status_type = -1; // return
            // status0: empty
            // status1: Turn Left
            // status2: Turn Right
            // status3: Curved point
            // status4: Straight Turn close
            // status5: Straight Turn long
            // status6: Straight
            std::vector<dpoints_t> v_xy;
            for(int lane_id=0; lane_id<v_src_dist_lane_points.size(); lane_id++){
                const dpoints_t& one_lane_points = v_src_dist_lane_points[lane_id];
                dpoints_t t_xy;
                dpoint_t t_x, t_y;
                // change x,y to y,x; t_x from small to large
                for(int idx=one_lane_points.size()-1; idx>0; idx--){
                    t_x.push_back(one_lane_points[idx][1]);
                    t_y.push_back(one_lane_points[idx][0]);
                }
                t_xy.push_back(t_x);
                t_xy.push_back(t_y);
                v_xy.push_back(t_xy);
            }

            //step1: cal one order error
            std::vector<double> y_error_list;
            for(int lane_id=0; lane_id<v_xy.size(); lane_id++){
                const dpoints_t& t_xy = v_xy[lane_id];
                dpoint_t t_x = t_xy[0], t_y = t_xy[1];
                std::vector<double> param_list_1 = polyfit(t_x, t_y, 1);
                double y_error = 0.0;
                for (int idx=0; idx<t_x.size(); idx++)
                    y_error += pow((t_y[idx] - polyfit_predict(param_list_1, t_x[idx])), 2);
                y_error_list.push_back(y_error);
            }

            double tmp_list[2];
            for(int lane_id=0; lane_id<v_xy.size(); lane_id++){
                    const dpoints_t& t_xy = v_xy[lane_id];
                    dpoint_t t_x = t_xy[0], t_y = t_xy[1];

                    std::vector<double> param_list_1;
                    // new distance_table
                    // -0.00144898 -0.77427519
                    // 0.0054114  0.67191294
                    if (lane_id == 0){
                        param_list_1.push_back(-0.00144898);
                        param_list_1.push_back(-0.77427519);
                    }
                    else if (lane_id == 1){
                        param_list_1.push_back(0.0054114);
                        param_list_1.push_back(0.67191294);
                    }


                    double y_error = 0.0;
                    for (int idx=0; idx<t_x.size(); idx++){
                        y_error += pow((t_y[idx] - polyfit_predict(param_list_1, t_x[idx])), 2);
                    }
                    tmp_list[lane_id] = fabs(y_error);
            }
            // std::cout << "tmp_list:" << tmp_list[0] << " " << tmp_list[1] << std::endl;
            // std::cout << "sub error:" << fabs(tmp_list[0]-tmp_list[1]) << std::endl;


            //step2: judge lane straight line / curved line
            // curved line
            // std::cout << "y_error_list:" << y_error_list[0] << " " << y_error_list[1] << std::endl;
            if (y_error_list[0] >= 1.0 || y_error_list[1] >= 1.0){
                // std::cout << "curved line" << std::endl;
                std::vector<double> lane_dist_list, y_error_total_list;
                for(int lane_id=0; lane_id<v_xy.size(); lane_id++){
                    const dpoints_t& t_xy = v_xy[lane_id];
                    dpoint_t t_x = t_xy[0], t_y = t_xy[1];

                    std::vector<double> param_list_1;
                    // old distance_table
                    // if (lane_id == 0){
                    //     param_list_1.push_back(-0.01333192);
                    //     param_list_1.push_back(-0.74828003);
                    // }
                    // else if (lane_id == 1){
                    //     param_list_1.push_back(-0.01084933);
                    //     param_list_1.push_back(0.749025);
                    // }

                    // new distance_table
                    // -0.00144898 -0.77427519
                    // 0.0054114  0.67191294
                    if (lane_id == 0){
                        param_list_1.push_back(-0.00144898);
                        param_list_1.push_back(-0.77427519);
                    }
                    else if (lane_id == 1){
                        param_list_1.push_back(0.0054114);
                        param_list_1.push_back(0.67191294);
                    }

                    int idx_0 = 0;
                    double y_error = 0.0;
                    for (int idx=0; idx<t_x.size(); idx++){
                        y_error = pow((t_y[idx] - polyfit_predict(param_list_1, t_x[idx])), 2);
                        // std::cout << t_y[idx] << " " << t_x[idx] << " " << polyfit_predict(param_list_1, t_x[idx]) << std::endl;
                        if (y_error > 0.1){
                            idx_0 = idx;
                            break;
                        }
                    }

                    double y_error_total = 0.0;
                    for (int idx=0; idx<int(t_x.size()*0.3); idx++)
                        y_error_total += (t_y[idx] - polyfit_predict(param_list_1, t_x[idx]));
                    y_error_total_list.push_back(y_error_total);

                    curved_point_list.push_back({t_x[idx_0], t_y[idx_0]});
                    lane_dist_list.push_back(t_y[0]);
                    double cur_R = GetCurvatureR(v_param_list[lane_id], t_x[idx_0]);
                    curved_r_list.push_back(cur_R);
                }//end for
                // judge curved line status
                double flag = lane_dist_list[0] + lane_dist_list[1];
                // std::cout << "flag:" << flag << std::endl;
                // std::cout << "y_error_total_list 0 :" << y_error_total_list[0] << std::endl;
                // std::cout << "y_error_total_list 1 :" << y_error_total_list[1] << std::endl;
                // std::cout << curved_r_list[0] << " " << curved_r_list[1] << std::endl;
                // std::cout << curved_point_list[0][0] << " " << curved_point_list[1][0] << std::endl;

                // old distance_table
                // if (fabs(flag) > 0.4){
                
                // new distance_table
                if (fabs(flag) > 0.25){
                    status_type = 0; // "status0"; // empty
                    if (y_error_total_list[0] < 0 && y_error_total_list[1] < 0){
                        status_type = 1; // "status1"; // Turn Left
                    }
                    else if (y_error_total_list[0] > 0 && y_error_total_list[1] > 0){
                        status_type = 2; // "status2"; // Turn Right
                    }
                }
                else{
                    if (curved_r_list[0]>=135 && curved_r_list[0]<=1000 && curved_r_list[1]>=135 && curved_r_list[1]<=1000 && (curved_point_list[0][0]-curved_point_list[1][0] < 10.0) ){
                            status_type = 3; // "status3"; // Curved point
                    }
                    else{
                        if (curved_point_list[0][0]<=30 && curved_point_list[1][0]<=30){
                            status_type = 4; // "status4"; // Straight Turn close
                        }
                        else{
                            status_type = 5; // "status5"; // Straight Turn long
                        }
                    }
                }// end if else
            }
            // straight line
            else{
                // std::cout << "straight line" << std::endl;
                std::vector<double> straight_error_list, lane_dist_list, y_error_total_list;
                for(int lane_id=0; lane_id<v_xy.size(); lane_id++){
                    const dpoints_t& t_xy = v_xy[lane_id];
                    dpoint_t t_x = t_xy[0], t_y = t_xy[1];

                    std::vector<double> param_list_1;
                    // old distance_table
                    // if (lane_id == 0){
                    //     param_list_1.push_back(-0.01333192);
                    //     param_list_1.push_back(-0.74828003);
                    // }
                    // else if (lane_id == 1){
                    //     param_list_1.push_back(-0.01084933);
                    //     param_list_1.push_back(0.749025);
                    // }

                    // new distance_table
                    // -0.00144898 -0.77427519
                    // 0.0054114  0.67191294
                    if (lane_id == 0){
                        param_list_1.push_back(-0.00144898);
                        param_list_1.push_back(-0.77427519);
                    }
                    else if (lane_id == 1){
                        param_list_1.push_back(0.0054114);
                        param_list_1.push_back(0.67191294);
                    }


                    double y_error = 0.0, y_error_total = 0.0;
                    for (int idx=0; idx<t_x.size(); idx++){
                        y_error += pow((t_y[idx] - polyfit_predict(param_list_1, t_x[idx])), 2);
                        y_error_total += (t_y[idx] - polyfit_predict(param_list_1, t_x[idx]));
                    }

                    straight_error_list.push_back(y_error);
                    y_error_total_list.push_back(y_error_total);
                    lane_dist_list.push_back(t_y[0]);
                }//end for
                // judge straight line status
                double flag = lane_dist_list[0] + lane_dist_list[1];
                // std::cout << "straight_error_list:" << straight_error_list[0] << " " << straight_error_list[1] << std::endl;
                // std::cout << (straight_error_list[0]<=5.0 || straight_error_list[1]<=5.0) << std::endl;
                if (straight_error_list[0]<=5.0 || straight_error_list[1]<=5.0){
                    status_type = 6; // "status6"; // Straight
                }
                else{
                    status_type = 0; // "status0"; // empty
                    if (y_error_total_list[0] < 0 && y_error_total_list[1] < 0){
                        status_type = 1; // "status1"; // Turn Left
                    }
                    else if (y_error_total_list[0] > 0 && y_error_total_list[1] > 0){
                        status_type = 2; // "status2"; // Turn Right
                    }
                }
            }// end if else

            return status_type;
        }// end GetLaneStatus

    }// namespace algorithm
}//namespace watrix
