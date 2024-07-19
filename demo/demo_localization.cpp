
#include "roadlib.h"
#include "gviewer.h"
#include "visualization.h"
#include <fstream>


gviewer viewer;
vector<VisualizedInstance> vis_instances;
std::normal_distribution<double> noise_distribution;
std::default_random_engine random_engine; 

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
	const pair<double,Tpoint>& initial_guess,
	Trajectory& traj_user, string result_file);

int main(int argc, char *argv[])
{
	if(argc!=9)//首先检查命令行参数数量是否正确
	{
		std::cerr<<"[ERROR] Number of arguments not correct!"<<std::endl;
		std::cerr<<"[USAGE] demo_mapping CONFIG_PATH IMAGE_TIMESTAMP_FILE IMAGE_DIR SEMANTIC_DIR REF_POSE_FILE ODO_POSE_FILE MAP_FILE RESULT_POSE_FILE"<<std::endl;
		exit(1);
	}
	string config_file(argv[1]);// = "D:/Projects/road_vision_iros/config/0412/vi.yaml";
	string stamp_file(argv[2]); //= "D:/city_0412/image/stamp_rearrange.txt";
	string raw_dir(argv[3]); //= "D:/city_0412/image/cam0";
	string semantic_dir(argv[4]); //= "D:/city_0412/image/out/pred";
	string ref_file(argv[5]); //= "D:/city_0412/gt.txt"
	string odo_file(argv[6]); //= "D:/city_0412/odo.txt";
	string map_file(argv[7]); //= "D:/city_0412/map.bin";
	string result_file(argv[8]); //= "D:/city_0412/result.txt";

	viewer.Show();

	SensorConfig config (config_file);

	Trajectory traj_gt = load_global_trajectory(ref_file); //加载全局轨迹（仅用于初始猜测）
	Trajectory traj_odo = load_local_trajectory(odo_file); //本地轨迹
	map<double, string> camstamp = load_camstamp(stamp_file);//加载相机时间戳映射

	// Load map.
	RoadInstancePatchMap road_map;//加载道路地图 road_map，并解冻地图以便后续使用
	road_map.loadMapFromFileBinaryRaw(map_file);
	road_map.unfreeze();
	
	// In this example, initial guess of the pose is needed.
    traj_gt.transfrom_ref(road_map.ref);//根据地图参考系将全局轨迹转换为地图参考系
	pair<double, Tpoint> initial_guess = make_pair(config.t_start, traj_gt.poses[config.t_start]);//并生成初始猜测的姿态
	std::cout<<"[INFO] initial guess: "<<setprecision(3)<<setiosflags(ios::fixed)<<config.t_start<<" "
		<<initial_guess.second.t.transpose()<<std::endl;
	
	// Map-aided localization.
	Trajectory traj_result;

	main_phase_localization(config, camstamp, raw_dir, semantic_dir,
		road_map, traj_odo, initial_guess, traj_result, result_file);

	return 0;
}
