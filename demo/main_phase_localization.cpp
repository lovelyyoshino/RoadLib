
#include <unordered_map>
#include "roadlib.h"
#include "gviewer.h"
#include "fgo.hpp"
#include <fstream>
#include "visualization.h"

/// @brief 这个函数是地图辅助定位的主函数，接收一系列参数，包括传感器配置、相机时间戳、原始图像目录、语义图像目录、道路实例补丁地图、用户视觉里程计轨迹、初始猜测、用户轨迹等
/// @param config 传感器配置
/// @param camstamp 相机时间戳映射
/// @param raw_dir 原始图像目录
/// @param semantic_dir 语义图像目录
/// @param road_map 道路实例补丁地图
/// @param traj_vio_user 用户视觉里程计轨迹
/// @param initial_guess 初始猜测
/// @param traj_user 用户轨迹
/// @param result_file 结果文件路径
/// @return 
int main_phase_localization(const SensorConfig& config,
	const map<double, string>& camstamp,
	const string& raw_dir, const string& semantic_dir,
	RoadInstancePatchMap& road_map,
	const Trajectory& traj_vio_user,
	const pair<double, Tpoint>& initial_guess,
	Trajectory& traj_user, string result_file)
{
	gv::IPMProcesser ipm_processer(config.cam, gv::IPMType::NORMAL);//初始化IPM处理器，输入的是相机配置和IPM类型

	ofstream ofs_result(result_file);
	road_map.buildKDTree();//建立道路地图的KD树

	std::cerr << "[ PHASE ] : Map-aided localization." << std::endl;

	// A pose window for smoothing.
	deque<pair<Eigen::Matrix3d, Eigen::Vector3d>> pose_history;//定义一个双端队列，用于存储姿态历史记录

	Vector3d graph_pos_ref = road_map.ref;//道路地图的参考位置
	Matrix3d graph_Rne_ref = calcRne(road_map.ref);//计算参考位置的东北旋转坐标

	map<double, shared_ptr<RoadInstancePatchFrame>> all_frames;//所有帧的指针
	map<double, long long> keyframes_flag;//关键帧标志，对应时间，id号
	long long keyframes_count=0;//关键帧计数，会传入keyframes_flag
	int frame_count = 0;
	Eigen::Vector3d last_att_ave;
	for (auto image_iter = camstamp.begin(); image_iter != camstamp.end(); image_iter++)//遍历相机时间戳映射
	{
		//对每一帧图像进行处理，根据时间戳判断是否在指定时间范围内
		if (image_iter->first < config.t_start) continue;
		if (image_iter->first > config.t_end) break;

		frame_count++;

		Eigen::Matrix3d R_c1c0 = config.cam.cg.getR();//获取相机的旋转矩阵

		// Get the latest IMU pose.
		auto this_pose = traj_vio_user.poses.lower_bound(image_iter->first - 0.001);//查找最接近当前图像时间戳的视觉里程计姿态
		if (fabs(this_pose->first - image_iter->first) > 0.001) continue; //在时间戳映射中找不到对应的视觉里程计姿态，跳过

		pose_history.push_back(make_pair(this_pose->second.R, this_pose->second.t));//将当前时刻的姿态信息加入历史记录队列
		while (pose_history.size() > config.pose_smooth_window)//如果历史记录队列的长度超过了设定的窗口大小，就将队列头部的元素删除
		{
			pose_history.pop_front();
		}

		if (config.need_smooth)//如果需要姿态平滑处理，则执行以下代码块
		{
			// Transformation between the smoothed IMU frame and the latest IMU frame.
			Eigen::Vector3d att_ave(0, 0, 0);
			Eigen::Vector3d att_diff;
			Eigen::Matrix3d R_b_n;
			Eigen::Vector3d att_this;
			for (int i = 0; i < pose_history.size(); i++)
			{
				R_b_n = pose_history[i].first;
				att_this = m2att(R_b_n);//将旋转矩阵转换为欧拉角
				att_ave += att_this;//计算历史记录队列中所有姿态的平均值
				if (i == pose_history.size() - 1)
				{
					att_ave /= pose_history.size();
					att_diff = att_this - att_ave;//计算当前姿态与平均姿态的差值
				}
			}

			if (fabs((att_ave(0) - last_att_ave(0)) * R2D) > config.large_slope_thresold / config.pose_smooth_window && pose_history.size() > 5) //如果当前姿态与平均姿态的差值超过了设定的阈值，就清空历史记录队列
			{
				std::cerr << "[INFO] Large slope detected!" << std::endl;
				pose_history.clear();
				pose_history.push_back(make_pair(this_pose->second.R, this_pose->second.t));//将当前时刻的姿态信息加入历史记录队列，重新平滑后面的
				last_att_ave = att_this;//更新上一次的平均姿态
				// int count_temp = 0;
				// for (auto iter_frame = all_frames.rbegin(); iter_frame != all_frames.rend() && count_temp < 20; iter_frame++)
				// {
				// 	Vector3d ti0i1 = iter_frame->second->R.transpose()*(this_pose->second.t - iter_frame->second->t);
				// 	// std::cout << "[INFO] ti0i1 : " << ti0i1.transpose() << std::endl;
				// 	if (ti0i1(1) < 13.5 && (ti0i1(0)) < 5)
				// 		for(auto iter_class = iter_frame->second->patches.begin();iter_class!=iter_frame->second->patches.end();iter_class++)
				// 			for(int ipatch = 0;ipatch<iter_class->second.size();ipatch++)
				// 			 iter_class->second[ipatch]->valid_add_to_map = false;
				// 	count_temp++;
				// }
			}
			else
			{
				last_att_ave = att_ave;
			}
			//	Transformation between the reference camera frame(IPM frame) and the latest camera frame.
			// 获取最新imu帧与平均imu帧之间的旋转矩阵
			Eigen::Matrix3d R_ilatest_iave = a2mat(Vector3d(att_diff(0) /*- att_ave(0) * 0.25*/, att_diff(1), 0.0));
			Eigen::MatrixXd R_clatest_cave = config.cam.Ric.transpose() * R_ilatest_iave * config.cam.Ric;//将imu帧的旋转矩阵转换为相机帧的旋转矩阵

			R_c1c0 = (R_c1c0.transpose() * R_clatest_cave).transpose();//将最新相机帧的旋转矩阵与历史记录队列中的旋转矩阵进行平滑处理
		}

		ipm_processer.updateCameraGroundGeometry(gv::CameraGroundGeometry(R_c1c0, config.cam.cg.getH()));//根据相机旋转矩阵和地面高度，更新相机的地面几何信息，用于后续的透视变换处理

		// 读取原始图像
		cv::Mat img_raw = cv::imread(raw_dir + "/" + image_iter->second);

		// 生成透视变换后的图像
		cv::Mat ipm = ipm_processer.genIPM(img_raw, true);

		// 读取语义图像
		cv::Mat img_semantic;
		if (image_iter->second.find("/") == string::npos)
			img_semantic = cv::imread(semantic_dir + "/" + image_iter->second.substr(0, image_iter->second.find(".")) + ".png", -1);
		else
			img_semantic = cv::imread(semantic_dir + "/" + image_iter->second.substr(image_iter->second.find("/"), image_iter->second.find(".") - image_iter->second.find("/")) + ".png", -1);

		cv::Mat img_semantic_temp = cv::Mat(img_raw.rows, img_raw.cols, CV_8UC1);
		img_semantic_temp.setTo(cv::Scalar(0));//初始化语义图像的临时图像
		img_semantic.copyTo(img_semantic_temp(cv::Rect(0, img_raw.rows - img_semantic.rows,
			img_raw.cols, img_semantic.rows)));//将语义图像拷贝到原始图像的底部
		img_semantic = img_semantic_temp;//更新语义图像
		cv::Mat ipm_semantic = ipm_processer.genIPM(img_semantic, true);//生成透视变换后的语义图像

		// Generate instance-level patches.
		auto this_frame_ptr = make_shared<RoadInstancePatchFrame>(generateInstancePatch(config, ipm_processer, ipm, ipm_semantic));//根据配置、透视变换处理器和透视变换后的图像，生成道路实例补丁帧，包括实例级别的补丁信息
		auto& this_frame = *this_frame_ptr;
		this_frame.R = this_pose->second.R; // At this moment, this pose is still odometry pose.
		this_frame.t = this_pose->second.t;
		this_frame.time = this_pose->first;
		this_frame.generateMetricPatches(config, ipm_processer);//生成度量级别的补丁信息

		
		int old_old_visualize_size;

		// 根据已有的帧数判断是否需要预测当前姿态，这是对上一帧结果进行处理分析的
		if(all_frames.size()>0)
		{
			auto iter_traj_vio_user = traj_vio_user.poses.lower_bound(all_frames.rbegin()->first - 0.001);//查找最接近当前图像时间戳的视觉里程计姿态，lower_bound是大于等于当前时刻
			auto iter_temp = all_frames.rbegin(); iter_temp++;//指向上一个元素，也就是后一个元素，这里是不是有问题？感觉是--？
			auto iter_traj_vio_user_before = traj_vio_user.poses.lower_bound(iter_temp->first - 0.001);//上一帧的位置信息
			double dist_from_last_keyframe = (iter_traj_vio_user->second.t - iter_traj_vio_user_before->second.t).norm();//计算两帧之间的距禈

			if (dist_from_last_keyframe <config.localization_min_keyframe_dist && all_frames.size()>1)
			{
				// std::cout << "[INFO] Skip keyframe: " << setprecision(3) << setiosflags(ios::fixed) << all_frames.rbegin()->first
				// 	<< " Travelled distance : " << setprecision(3) << setiosflags(ios::fixed) << dist_from_last_keyframe << std::endl;
				all_frames.erase(all_frames.rbegin()->first);//删除最后一帧
			}
			else
			{
				keyframes_flag[all_frames.rbegin()->first] = keyframes_count++;//更新关键帧标志
			}
		}

		all_frames.insert(make_pair(image_iter->first, this_frame_ptr));//将当前帧加入帧队列
		keyframes_flag[all_frames.rbegin()->first] = -1;//更新关键帧标志

		while (all_frames.size() > config.localization_max_windowsize)//如果帧队列的长度超过了设定的窗口大小，就删除队列头部的元素
		{
			all_frames.erase(all_frames.begin());
		}

		// Predict current pose based on last pose estimation and relative pose.
		// 根据上一帧的姿态估计和相对姿态预测当前姿态
		if ( all_frames.size() > 0)
		{
			auto iter_frame = all_frames.rbegin();
			auto iter_frame_before = all_frames.rbegin();
			iter_frame_before--;//指向下一个元素，也就是前一个元素

			auto iter_traj_vio_user = traj_vio_user.poses.lower_bound(iter_frame->first - 0.001);
			auto iter_traj_vio_user_before = traj_vio_user.poses.lower_bound(iter_frame_before->first - 0.001);
			if (iter_traj_vio_user != traj_vio_user.poses.end() && fabs(iter_traj_vio_user->first - iter_frame->first) < 0.001
				&& iter_traj_vio_user_before != traj_vio_user.poses.end() && fabs(iter_traj_vio_user_before->first - iter_frame_before->first) < 0.001)
			{
				Vector3d t0 = iter_traj_vio_user_before->second.t;
				Matrix3d R0 = iter_traj_vio_user_before->second.R;
				Vector3d t1 = iter_traj_vio_user->second.t;
				Matrix3d R1 = iter_traj_vio_user->second.R;

				Matrix3d R01 = R0.transpose() * R1;//计算两帧之间的旋转矩阵
				Quaterniond q01 = Quaterniond(R01).normalized();//将旋转矩阵转换为四元数
				Vector3d t01 = R0.transpose() * (t1 - t0);//计算两帧之间的平移向量

				//根据VIO轨迹以及上一帧位置推算当前位置
				(*iter_frame->second).t = (*iter_frame_before->second).t + (*iter_frame_before->second).R * t01;
				(*iter_frame->second).R = (*iter_frame_before->second).R * R01;
			}
		}

		// 根据配置判断是否需要进行图像可视化，如果需要则展示透视变换后的图像和语义图像
		if (config.enable_vis_image)
		{
			cv::Mat ipm_color(ipm_semantic.rows, ipm_semantic.cols, CV_8UC3);
			ipm_color.setTo(cv::Vec3b(0, 0, 0));
			genColorLabel(ipm_semantic, ipm_color);//生成语义图像的颜色标签
			cv::Mat ipm_semantic_cluster = cv::Mat(config.cam.IPM_HEIGHT, config.cam.IPM_WIDTH, CV_8UC3);
			ipm_color.copyTo(ipm_semantic_cluster);

			cv::imshow("ipm", ipm);
			cv::imshow("ipm_semantic_cluster", ipm_semantic_cluster);
			cv::waitKey(1);
		}

		vector<vector<Vector3d>> vis_points_this_frame;
		vector<VisualizedInstance> vis_instances;

		// 根据配置判断是否需要进行三维可视化，如果需要则展示道路地图和当前帧的点云
		if (config.enable_vis_3d)
		{
			visualize_roadmap(road_map, vis_instances);//可视化道路地图
			viewer.SetInstances(vis_instances);//设置可视化实例

			// Pre-iteration visualization.
			for (auto iter_class = this_frame.patches.begin(); iter_class != this_frame.patches.end(); iter_class++)
			{
				vis_points_this_frame.push_back(vector<Vector3d>());
				for (auto iter_instance = iter_class->second.begin(); iter_instance != iter_class->second.end(); iter_instance++)
				{
					for (auto& p : (*iter_instance)->points_metric) vis_points_this_frame.back().push_back(this_frame.t + this_frame.R * p);//将当前帧的点云加入到可视化队列
				}
			}
			viewer.SetPointCloudSemantic(vis_points_this_frame);
			viewer.SetFrames(vector<pair<Matrix3d, Vector3d>>(1, make_pair(this_frame.R, this_frame.t)));
		}

		// 对当前帧进行多次迭代优化，包括姿态优化、地图匹配、参数更新等过程
		for (int i_iter = 0; i_iter < 3; i_iter++)
		{
			std::vector<ceres::ResidualBlockId> psrIDs;
			ceres::Problem problem;
			ceres::Solver::Options options;
			ceres::Manifold* local_parameterization = new ceres::QuaternionManifold();
			vector<t_graphpose> all_graph_poses;
			map<pair<long long, int>, t_patch_est> map_patch_est; // All the patches in the map that need to be updated.

			// 清楚所有的可视化实例，用于重新绘制
			if (i_iter > 0)
			{
				vis_instances.resize(old_old_visualize_size);
			}
			old_old_visualize_size = vis_instances.size();

			int ccount = 0;
			
			// 对每一帧进行地图匹配，包括点到线的匹配，更新姿态和地图信息
			for (auto iter_frame = all_frames.begin(); iter_frame != all_frames.end(); iter_frame++, ccount++)
			{
				// 检查是否可以匹配，条件是按照关键帧的间隔或者到达最后几帧
				bool enable_match = (keyframes_flag[iter_frame->first] % config.localization_every_n_frames ==0 || ccount > all_frames.size() - config.localization_force_last_n_frames);
				map<PatchType, vector<pair<int, int>>> class_pairs;

				map<PatchType, map<int, pair<int, vector<pair<int, int>>>>> line_pairs;//线到线的匹配，这里存放的是：检测类型，当前帧的线的id，地图中的线的id，匹配的点的id

				auto& this_graph_frame = (*iter_frame->second);//当前帧的道路实例补丁帧
				if (enable_match)
				{
					if (i_iter > 0)//这个的作用好像没有？
						class_pairs = road_map.mapMatch(config, this_graph_frame, 1);//地图匹配，查看当前帧的道路实例补丁帧是否与地图中的道路实例补丁匹配
					else
						class_pairs = road_map.mapMatch(config, this_graph_frame, 1);

					vector<PatchType> line_class_list = { PatchType::SOLID,PatchType::STOP };
					for (auto line_class : line_class_list)//对于实线和停车线，进行线到线的匹配
					{
						line_pairs[line_class] = map<int, pair<int, vector<pair<int, int>>>>();
						for (auto iter_line_match = class_pairs[line_class].begin(); iter_line_match != class_pairs[line_class].end(); iter_line_match++)
						{
							int frame_line_count = iter_line_match->second;//当前帧的线的id
							int map_line_count = iter_line_match->first;//地图中的线的id
							if (i_iter > 0)
								line_pairs[line_class].emplace(frame_line_count, make_pair(map_line_count, road_map.getLineMatch(config, this_graph_frame, line_class,
									frame_line_count, map_line_count, 1)));//线到线的匹配,这里存放的是：检测类型，当前帧的线的id，地图中的线的id，匹配的点的id
							else
								line_pairs[line_class].emplace(frame_line_count, make_pair(map_line_count, road_map.getLineMatch(config, this_graph_frame, line_class,
									frame_line_count, map_line_count, 1)));
						}

#ifdef PRINT_INFO
						for (int i = 0; i < line_pairs[frame_line_count].second.size(); i++)
							std::cerr << line_pairs[frame_line_count].second[i].first << " " << line_pairs[frame_line_count].second[i].second << std::endl;
#endif
					}
				}
				// 如果匹配到的四种特征都不满足，就清空匹配结果。这四种特征是虚线、引导线、停车线和实线
				if (class_pairs[PatchType::DASHED].size() == 0 && class_pairs[PatchType::GUIDE].size() == 0
					&& class_pairs[PatchType::STOP].size() == 0 && class_pairs[PatchType::SOLID].size() <= 2)
				{
					class_pairs.clear();
					line_pairs.clear();
				}

				// 将当前帧的姿态信息加入到优化问题中
				Quaterniond this_graph_frame_q = Quaterniond(this_graph_frame.R).normalized();

				double* p3 = new double[3];
				p3[0] = this_graph_frame.t(0);
				p3[1] = this_graph_frame.t(1);
				p3[2] = this_graph_frame.t(2);

				double* p4 = new double[4];
				p4[0] = this_graph_frame_q.w();
				p4[1] = this_graph_frame_q.x();
				p4[2] = this_graph_frame_q.y();
				p4[3] = this_graph_frame_q.z();
				//完成姿态信息的加入，这个是对应关联信息，这里面是初始的一些位置信息
				all_graph_poses.push_back({ iter_frame->first ,this_graph_frame.R,this_graph_frame.t,p3,p4 });
				problem.AddParameterBlock(all_graph_poses.back().q_array, 4, local_parameterization);//将姿态信息加入到优化问题中
				problem.AddParameterBlock(all_graph_poses.back().t_array, 3);//将位置信息加入到优化问题中
		
				// 加入landmark参数，这些参数是固定的
				for (auto iter_class = class_pairs.begin(); iter_class != class_pairs.end(); iter_class++)
				{
					if (iter_class->first == PatchType::DASHED || iter_class->first == PatchType::GUIDE)//对于虚线和引导线，加入四个边界点的参数
					{
						for (int j = 0; j < iter_class->second.size(); j++)//对应的是当前帧的虚线和引导线对应的四个点信息
						{
							auto& this_patch_map = road_map.patches[iter_class->first][iter_class->second[j].first];//获取对应类型的地图信息
							if (map_patch_est.find(make_pair(this_patch_map->id, 0)) != map_patch_est.end()) continue;//如果已经加入了，就跳过

							// 对于补丁实例，将四个边界点加入到参数中，这里需要将新的特征加入到地图当中
							for (int iii = 0; iii < 4; iii++)
							{
								map_patch_est[make_pair(this_patch_map->id, iii)] = t_patch_est(this_patch_map->b_point_metric[iii]);//将四个边界点加入到参数中
								problem.AddParameterBlock(map_patch_est[make_pair(this_patch_map->id, iii)].p, 3);//将边界点的位置信息加入到优化问题中
								problem.SetParameterBlockConstant(map_patch_est[make_pair(this_patch_map->id, iii)].p);//将边界点的位置信息设置为固定的，因为这些是一些固定的特征点
							}
#ifdef PRINT_INFO
							std::cerr << "Add Dot patch : " << this_patch_map->id << " " << 0 << std::endl;
#endif
						}
					}
					else if (iter_class->first == PatchType::SOLID || iter_class->first == PatchType::STOP)//对于实线和停车线，加入线段的参数
					{
						for (int j = 0; j < iter_class->second.size(); j++)
						{
							auto& this_patch_map = road_map.patches[iter_class->first][iter_class->second[j].first];//当前地图当中的id信息

							for (int k = 0; k < line_pairs[iter_class->first][iter_class->second[j].second].second.size(); k++)//遍历当前帧的线id当中的所有匹配点
							{
								int instance_vertex_map = line_pairs[iter_class->first][iter_class->second[j].second].second[k].second;//地图中的线的id
								for (int zzz = 0; zzz < 2; zzz++)
									if (map_patch_est.find(make_pair(this_patch_map->id, instance_vertex_map + zzz)) == map_patch_est.end())//如果没有加入到地图当中
									{
										map_patch_est[make_pair(this_patch_map->id, instance_vertex_map + zzz)] = t_patch_est(this_patch_map->line_points_metric[instance_vertex_map + zzz]);
										problem.AddParameterBlock(map_patch_est[make_pair(this_patch_map->id, instance_vertex_map + zzz)].p, 3);//将线段的位置信息加入到优化问题中
										problem.SetParameterBlockConstant(map_patch_est[make_pair(this_patch_map->id, instance_vertex_map + zzz)].p);
#ifdef PRINT_INFO
										std::cerr << "Add Line patch : " << this_patch_map->id << " " << instance_vertex_map + zzz << std::endl;
#endif
									}
							}
						}
					}
				}

				// Initial guess factor (ceres).
				// 初始化猜测因子
				if (fabs(iter_frame->first - initial_guess.first) < 0.001)//如果当前帧的时间戳与初始猜测的时间戳相同
				{
					ceres::LossFunction* gnss_loss_function = new ceres::TrivialLoss();//初始化猜测的损失函数，使用的是TrivialLoss，这是简单的损失函数，不对残差进行额外处理
					Vector3d t0 = initial_guess.second.t;
					Matrix3d R0 = initial_guess.second.R;
					Quaterniond q0 = Quaterniond(R0).normalized();
					ceres::CostFunction* gnss_pose_factor;//初始化猜测因子
					gnss_pose_factor = PoseVarError::Create(t0.x(), t0.y(), t0.z(),
						q0.w(), q0.x(), q0.y(), q0.z(), 10, 10, 10, 0.01, 0.01, 0.04);//创建姿态误差因子
					auto ID = problem.AddResidualBlock(gnss_pose_factor, gnss_loss_function, all_graph_poses.back().q_array,
						all_graph_poses.back().t_array);//将姿态误差因子加入到优化问题中,优化对应的all_graph_poses
					psrIDs.push_back(ID);
				}

				// Relative pose factor (ceres).
				// 相对姿态因子
				if (iter_frame != all_frames.begin())
				{
					auto iter_frame_before = iter_frame;
					iter_frame_before--;//指向上一个元素，也就是前一个元素

					auto iter_traj_vio_user = traj_vio_user.poses.lower_bound(iter_frame->first - 0.001);
					auto iter_traj_vio_user_before = traj_vio_user.poses.lower_bound(iter_frame_before->first - 0.001);
					if (iter_traj_vio_user != traj_vio_user.poses.end() && fabs(iter_traj_vio_user->first - iter_frame->first) < 0.001
						&& iter_traj_vio_user_before != traj_vio_user.poses.end() && fabs(iter_traj_vio_user_before->first - iter_frame_before->first) < 0.001)
					{
						ceres::LossFunction* vio_loss_function = new ceres::TrivialLoss();
						// ceres::LossFunction* vio_loss_function = new ceres::HuberLoss(1.0);

						Vector3d t0 = iter_traj_vio_user_before->second.t;
						Matrix3d R0 = iter_traj_vio_user_before->second.R;
						Vector3d t1 = iter_traj_vio_user->second.t;
						Matrix3d R1 = iter_traj_vio_user->second.R;

						Matrix3d R01 = R0.transpose() * R1;
						Quaterniond q01 = Quaterniond(R01).normalized();
						Vector3d t01 = R0.transpose() * (t1 - t0);

						double dist = t01.norm();
						dist = dist < 0.1 ? 0.1 : dist;//最小距离为0.1
						ceres::CostFunction* relative_pose_factor = RelativeRTVarError::Create(t01.x(), t01.y(), t01.z(),
							q01.w(), q01.x(), q01.y(), q01.z(),
							0.05 * dist, 0.1 * dist, 0.05 * dist,
							0.01 , 0.01, 0.01);//创建相对姿态因子

						problem.AddResidualBlock(relative_pose_factor, vio_loss_function, (all_graph_poses.rbegin() + 1)->q_array,
							(all_graph_poses.rbegin() + 1)->t_array,
							all_graph_poses.rbegin()->q_array,
							all_graph_poses.rbegin()->t_array);
					}
				}

				// Map matching (point-to-line) factor (ceres).
				ceres::LossFunction* map_loss_function = new ceres::HuberLoss(1.0);
				VisualizedInstance vis_instance;
				vis_instance.type = VisualizedPatchType::LINE_SEGMENT;
				vis_instance.pts = vector<Vector3d>(3);
				vis_instance.pts_color = vector<Vector3d>(3, Eigen::Vector3d(1.0, 0, 1.0));
				vis_instance.alpha = 1.0f;
				vis_instance.linewidth = 1.0f;

				for (auto iter_class = class_pairs.begin(); iter_class != class_pairs.end(); iter_class++)
				{
					for (int j = 0; j < iter_class->second.size(); j++)
					{
						auto& this_patch_map = road_map.patches[iter_class->first][iter_class->second[j].first];
						auto& this_patch_frame = this_graph_frame.patches[iter_class->first][iter_class->second[j].second];

						if (iter_class->first == PatchType::DASHED || iter_class->first == PatchType::GUIDE)
						{
							for (int k = 0; k < 4; k++)
							{
								Eigen::Vector3d pos_meas = (this_patch_frame->b_point_metric[k % 4] + this_patch_frame->b_point_metric[(k + 1) % 4])/2;
								ceres::CostFunction* map_factor = MapLineToFramePointFactor::Create(pos_meas.x(), pos_meas.y(), pos_meas.z(),
									this_patch_frame->b_unc_dist[k],
									this_patch_frame->b_unc_dist[k],
									this_patch_frame->b_unc_dist[k]);
								auto ID = problem.AddResidualBlock(map_factor, map_loss_function, all_graph_poses.back().q_array,
									all_graph_poses.back().t_array,
									map_patch_est[make_pair(this_patch_map->id, k % 4)].p,
									map_patch_est[make_pair(this_patch_map->id, (k + 1) % 4)].p);
								psrIDs.push_back(ID);
								pos_meas = this_graph_frame.R * pos_meas + this_graph_frame.t;
								vis_instance.pts[0] = Vector3d(map_patch_est[make_pair(this_patch_map->id, k % 4)].p);
								vis_instance.pts[1] = pos_meas;
								vis_instance.pts[2] = Vector3d(map_patch_est[make_pair(this_patch_map->id, (k+1) % 4)].p);
								vis_instances.push_back(vis_instance);
							}
						}
						else if (iter_class->first == PatchType::SOLID || iter_class->first == PatchType::STOP)
						{
							for (int k = 0; k < line_pairs[iter_class->first][iter_class->second[j].second].second.size(); k++)
							{
								int instance_vertex_frame = line_pairs[iter_class->first][iter_class->second[j].second].second[k].first;
								int instance_vertex_map = line_pairs[iter_class->first][iter_class->second[j].second].second[k].second;

								Vector3d pos_meas = this_patch_frame->line_points_metric[instance_vertex_frame];
								double uncertainty = sqrt(this_patch_frame->line_points_uncertainty[instance_vertex_frame](0, 0) +
									                      this_patch_frame->line_points_uncertainty[instance_vertex_frame](1, 1) +
									                      this_patch_frame->line_points_uncertainty[instance_vertex_frame](2, 2))/sqrt(3.0);
								ceres::CostFunction* map_factor = MapLineToFramePointFactor::Create(pos_meas.x(), pos_meas.y(), pos_meas.z(), 
									uncertainty,
									uncertainty,
									uncertainty);//自定义的残差项
								auto ID = problem.AddResidualBlock(map_factor, map_loss_function, all_graph_poses.back().q_array,
									all_graph_poses.back().t_array,
									map_patch_est[make_pair(this_patch_map->id, instance_vertex_map)].p,
									map_patch_est[make_pair(this_patch_map->id, instance_vertex_map + 1)].p);
								psrIDs.push_back(ID);//将添加的残差项的 ID 存储到 psrIDs 中
								pos_meas = this_graph_frame.R * pos_meas + this_graph_frame.t;//更新相机观测点的位置，将其从相机坐标系变换到世界坐标系
								vis_instance.pts[0] = Vector3d(map_patch_est[make_pair(this_patch_map->id, instance_vertex_map)].p);//将地图上的第一个点的位置存储到可视化数据结构中
								vis_instance.pts[1] = pos_meas;//将更新后的相机观测点的位置存储到可视化数据结构中
								vis_instance.pts[2] = Vector3d(map_patch_est[make_pair(this_patch_map->id, instance_vertex_map + 1)].p);//将地图上的第二个点的位置存储到可视化数据结构中
								vis_instances.push_back(vis_instance);
							}
						}

					}
				}
			}


			// Per-iteration visualization.
			if (config.enable_vis_3d)
			{
				ccount = 0;
				// 对于每一帧，如果开启了3D可视化，则进行可视化
				for (auto iter_frame = all_frames.begin(); iter_frame != all_frames.end(); iter_frame++, ccount++)
				{
					if (fabs(iter_frame->first - (all_frames.rbegin()->first)) < 0.001)
					{
						VisualizedInstance vis_instance_vehicle;
						vis_instance_vehicle.type = VisualizedPatchType::BOX;
						vis_instance_vehicle.color[0] = 0.0;
						vis_instance_vehicle.color[1] = 0.0;
						vis_instance_vehicle.color[2] = 0.0;
						vis_instance_vehicle.t = iter_frame->second->t;
						vis_instance_vehicle.R = a2mat(m2att(iter_frame->second->R));
						vis_instance_vehicle.h = 1.753;
						vis_instance_vehicle.l = 1.849;
						vis_instance_vehicle.w = 4.690;
						vis_instances.push_back(vis_instance_vehicle);
					}
					else if (keyframes_flag[iter_frame->first] % config.localization_every_n_frames == 0 || (ccount > all_frames.size() - config.localization_force_last_n_frames))
					{
						VisualizedInstance vis_instance_vehicle;
						vis_instance_vehicle.type = VisualizedPatchType::BOX;
						vis_instance_vehicle.color[0] = 0.8;
						vis_instance_vehicle.color[1] = 0.8;
						vis_instance_vehicle.color[2] = 0.8;
						vis_instance_vehicle.t = iter_frame->second->t;
						vis_instance_vehicle.R = a2mat(m2att(iter_frame->second->R));
						vis_instance_vehicle.h = 1.753;
						vis_instance_vehicle.l = 1.849;
						vis_instance_vehicle.w = 4.690;
						vis_instances.push_back(vis_instance_vehicle);
					}
				}
				viewer.SetInstances(vis_instances);
			}

			options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);//对上面构建的残差进行优化

			// Update poses.
			int frame_i = 0;
			for (auto iter_frame = all_frames.begin(); iter_frame != all_frames.end(); iter_frame++, frame_i++)
			{
				iter_frame->second->t = Vector3d(all_graph_poses[frame_i].t_array[0],
					all_graph_poses[frame_i].t_array[1],
					all_graph_poses[frame_i].t_array[2]);
				iter_frame->second->R = Quaterniond(all_graph_poses[frame_i].q_array[0],
					all_graph_poses[frame_i].q_array[1],
					all_graph_poses[frame_i].q_array[2],
					all_graph_poses[frame_i].q_array[3]).toRotationMatrix();
				delete[] all_graph_poses[frame_i].t_array;
				delete[] all_graph_poses[frame_i].q_array;
			}
		}

		// 优化过后的3D可视化
		if (config.enable_vis_3d)
		{
			int ccount = 0;
			for (auto iter_frame = all_frames.begin(); iter_frame != all_frames.end(); iter_frame++, ccount++)
			{
				if (fabs(iter_frame->first - (all_frames.rbegin()->first)) < 0.001)
				{
					VisualizedInstance vis_instance_vehicle;
					vis_instance_vehicle.type = VisualizedPatchType::BOX;
					vis_instance_vehicle.color[0] = 0.0;
					vis_instance_vehicle.color[1] = 0.0;
					vis_instance_vehicle.color[2] = 0.0;
					vis_instance_vehicle.t = iter_frame->second->t;
					vis_instance_vehicle.R = a2mat(m2att(iter_frame->second->R));
					vis_instance_vehicle.h = 1.753;
					vis_instance_vehicle.l = 1.849;
					vis_instance_vehicle.w = 4.690;
					vis_instances.push_back(vis_instance_vehicle);
				}
				else if (keyframes_flag[iter_frame->first] % config.localization_every_n_frames == 0 || (ccount > all_frames.size() - config.localization_force_last_n_frames))
				{
					VisualizedInstance vis_instance_vehicle;
					vis_instance_vehicle.type = VisualizedPatchType::BOX;
					vis_instance_vehicle.color[0] = 0.8;
					vis_instance_vehicle.color[1] = 0.8;
					vis_instance_vehicle.color[2] = 0.8;
					vis_instance_vehicle.t = iter_frame->second->t;
					vis_instance_vehicle.R = a2mat(m2att(iter_frame->second->R));
					vis_instance_vehicle.h = 1.753;
					vis_instance_vehicle.l = 1.849;
					vis_instance_vehicle.w = 4.690;
					vis_instances.push_back(vis_instance_vehicle);
				}

			}

			viewer.SetInstances(vis_instances);
			vis_points_this_frame.clear();
			for (auto iter_class = this_frame.patches.begin(); iter_class != this_frame.patches.end(); iter_class++)
			{
				vis_points_this_frame.push_back(vector<Vector3d>());
				for (auto iter_instance = iter_class->second.begin(); iter_instance != iter_class->second.end(); iter_instance++)
				{
					for (auto& p : (*iter_instance)->points_metric) vis_points_this_frame.back().push_back(this_frame.t + this_frame.R * p);
				}
			}
			viewer.SetPointCloudSemantic(vis_points_this_frame);
		}

		//如果开启了3D可视化
		if (ofs_result.good())
		{
			Eigen::Vector3d pos = (graph_pos_ref + graph_Rne_ref.transpose() * all_frames.rbegin()->second->t);
			Eigen::Quaterniond qnb  = Eigen::Quaterniond(calcRne(pos) * graph_Rne_ref.transpose() * all_frames.rbegin()->second->R);

			ofs_result << setprecision(3) << setiosflags(ios::fixed) << all_frames.rbegin()->second->time << " "
				<< setprecision(3) << setiosflags(ios::fixed) << pos(0) << " "
				<< setprecision(3) << setiosflags(ios::fixed) << pos(1) << " "
				<< setprecision(3) << setiosflags(ios::fixed) << pos(2) << " "
				<< setprecision(10) << setiosflags(ios::fixed) << qnb.w() << " "
				<< setprecision(10) << setiosflags(ios::fixed) << qnb.x() << " "
				<< setprecision(10) << setiosflags(ios::fixed) << qnb.y() << " "
				<< setprecision(10) << setiosflags(ios::fixed) << qnb.z() << std::endl;
		}

		std::cout << "[INFO] Map-aided localization: (frame) " << frame_count << " " 
			<<"(time) "<< setprecision(3) << setiosflags(ios::fixed) << all_frames.rbegin()->first << std::endl;

		//viewer.ScreenShot(image_iter->second.substr(image_iter->second.find("/")+1, image_iter->second.find(".") - image_iter->second.find("/")-1) + ".png");

	}

	return 0;
}