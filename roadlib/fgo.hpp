#pragma once

#include <ctime>
#include <chrono>
#include <cstdlib>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <queue>
#include <iostream>
#include <string>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>

/**
* @brief 计算给定四元数的逆四元数。逆四元数的实部与原始四元数相同，虚部取负值
* @param[in] q[4]				原始四元数
* @param[in] q_inverse[4]		逆四元数
*/
template <typename T> inline
void QuaternionInverse(const T q[4], T q_inverse[4])
{
	q_inverse[0] = q[0];
	q_inverse[1] = -q[1];
	q_inverse[2] = -q[2];
	q_inverse[3] = -q[3];
};


struct RelativeRTError
{
	/// @brief RelativeRTError结构体用于构建相对位姿变换误差因子
	/// @param t_x 平移向量x
	/// @param t_y 平移向量y
	/// @param t_z 平移向量z
	/// @param q_w 四元数
	/// @param q_x 四元数
	/// @param q_y 四元数
	/// @param q_z 四元数
	/// @param t_var 平移的方差
	/// @param q_var 旋转的方差
	RelativeRTError(double t_x, double t_y, double t_z,
		double q_w, double q_x, double q_y, double q_z,
		double t_var, double q_var)
		:t_x(t_x), t_y(t_y), t_z(t_z),
		q_w(q_w), q_x(q_x), q_y(q_y), q_z(q_z),
		t_var(t_var), q_var(q_var) {}

	template <typename T>
	bool operator()(const T* const w_q_i, const T* ti, const T* w_q_j, const T* tj, T* residuals) const
	{
		//根据输入的位姿信息，计算两个位姿之间的相对变换，在w坐标系下
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		T i_q_w[4];
		QuaternionInverse(w_q_i, i_q_w);

		T t_i_ij[3];
		ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);//使用四元数逆旋转t_w_ij到t_i_ij下，

		residuals[0] = (t_i_ij[0] - T(t_x)) / T(t_var);//然后计算平移的误差
		residuals[1] = (t_i_ij[1] - T(t_y)) / T(t_var);
		residuals[2] = (t_i_ij[2] - T(t_z)) / T(t_var);

		T relative_q[4];
		relative_q[0] = T(q_w);
		relative_q[1] = T(q_x);
		relative_q[2] = T(q_y);
		relative_q[3] = T(q_z);

		T q_i_j[4];
		ceres::QuaternionProduct(i_q_w, w_q_j, q_i_j);//计算两个四元数的乘积

		T relative_q_inv[4];
		QuaternionInverse(relative_q, relative_q_inv);

		T error_q[4];
		ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);//计算误差四元数

		residuals[3] = T(2) * error_q[1] / T(q_var);//然后计算旋转的误差
		residuals[4] = T(2) * error_q[2] / T(q_var);
		residuals[5] = T(2) * error_q[3] / T(q_var);

		return true;
	}

	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
		const double q_w, const double q_x, const double q_y, const double q_z,
		const double t_var, const double q_var)
	{
		return (new ceres::AutoDiffCostFunction<
			RelativeRTError, 6, 4, 3, 4, 3>(// 使用自动求导，将定义的代价函数结构体传入。第一个参数是代价函数的维度，第二个参数是待优化参数的维度（operator的维度）
				new RelativeRTError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, t_var, q_var)));
	}

	double t_x = 0, t_y = 0, t_z = 0, t_norm = 0;
	double q_w = 0, q_x = 0, q_y = 0, q_z = 0;
	double t_var = 0, q_var = 0;

};

struct MapPointSelfFactor
{
	/// @brief 构建地图点自身误差因子
	/// @param w_t_p_x 地图点在世界坐标系下的位置
	/// @param w_t_p_y 
	/// @param w_t_p_z 
	/// @param tx_var 位置的方差
	/// @param ty_var 
	/// @param tz_var 
	MapPointSelfFactor(double w_t_p_x, double w_t_p_y, double w_t_p_z,
		double tx_var, double ty_var, double tz_var)
		: w_t_p_x(w_t_p_x), w_t_p_y(w_t_p_y), w_t_p_z(w_t_p_z),
		tx_var(tx_var), ty_var(ty_var), tz_var(tz_var) {}

	double w_t_p_x = 0, w_t_p_y = 0, w_t_p_z = 0;
	double tx_var = 0, ty_var = 0, tz_var = 0;
	template <typename T>
	bool operator()(const T* w_t_p, T* residuals) const
	{
		residuals[0] = (w_t_p[0] - T(w_t_p_x))/T(tx_var);//计算地图点在世界坐标系下的位置与输入位置的差异
		residuals[1] = (w_t_p[1] - T(w_t_p_y))/T(ty_var);
		residuals[2] = (w_t_p[2] - T(w_t_p_z))/T(tz_var);
		return true;
	}
	static ceres::CostFunction* Create(const double w_t_p_x, const double w_t_p_y, const double w_t_p_z,
		const double tx_var, const double ty_var, const double tz_var)
	{
		return (new ceres::AutoDiffCostFunction<
			MapPointSelfFactor, 3, 3>(
				new MapPointSelfFactor(w_t_p_x, w_t_p_y, w_t_p_z,
					tx_var, ty_var, tz_var)));
	}
};

struct MapPointToFramePointFactor
{
	/// @brief 构建地图点到相机坐标系下点的误差因子
	/// @param i_t_p_x 相机坐标系下点的位置
	/// @param i_t_p_y 
	/// @param i_t_p_z 
	/// @param tx_var 位置的方差
	/// @param ty_var 
	/// @param tz_var 
	MapPointToFramePointFactor(double i_t_p_x, double i_t_p_y, double i_t_p_z,
		double tx_var, double ty_var, double tz_var)
		: i_t_p_x(i_t_p_x), i_t_p_y(i_t_p_y), i_t_p_z(i_t_p_z),
		tx_var(tx_var), ty_var(ty_var), tz_var(tz_var) {}

	double i_t_p_x = 0, i_t_p_y = 0, i_t_p_z = 0;
	double tx_var = 0, ty_var = 0, tz_var = 0;

	template <typename T>
	bool operator()(const T* const w_q_i, const T* const w_t_i,  const T* w_t_p, T* residuals) const
	{
		T w_t_ip[3];
		w_t_ip[0] = w_t_p[0] - w_t_i[0];
		w_t_ip[1] = w_t_p[1] - w_t_i[1];
		w_t_ip[2] = w_t_p[2] - w_t_i[2];
		T i_q_w[4];
		QuaternionInverse(w_q_i, i_q_w);

		T i_t_ip[3];
		ceres::QuaternionRotatePoint(i_q_w, w_t_ip, i_t_ip);//计算地图点在相机坐标系下的位置与输入位置的差异

		residuals[0] = (i_t_ip[0] - T(i_t_p_x)) / T(tx_var);
		residuals[1] = (i_t_ip[1] - T(i_t_p_y)) / T(ty_var);
		residuals[2] = (i_t_ip[2] - T(i_t_p_z)) / T(tz_var);

		return true;

	}
	static ceres::CostFunction* Create(const double i_t_p_x, const double i_t_p_y, const double i_t_p_z,
		const double tx_var, const double ty_var, const double tz_var)
	{
		return (new ceres::AutoDiffCostFunction<
			MapPointToFramePointFactor, 3, 4, 3, 3>(
				new MapPointToFramePointFactor(i_t_p_x, i_t_p_y, i_t_p_z,
					tx_var, ty_var, tz_var)));
	}
};

struct MapLineToFramePointFactor
{
	/// @brief 构建地图线到相机坐标系下点的误差因子
	/// @param i_t_p_x 相机坐标系下点的位置
	/// @param i_t_p_y 
	/// @param i_t_p_z 
	/// @param tx_var 位置的方差
	/// @param ty_var 
	/// @param tz_var 
	MapLineToFramePointFactor(double i_t_p_x, double i_t_p_y, double i_t_p_z,
		double tx_var, double ty_var, double tz_var)
		: i_t_p_x(i_t_p_x), i_t_p_y(i_t_p_y), i_t_p_z(i_t_p_z),
		tx_var(tx_var), ty_var(ty_var), tz_var(tz_var) {}

	double i_t_p_x = 0, i_t_p_y = 0, i_t_p_z = 0;
	double tx_var = 0, ty_var = 0, tz_var = 0;

	template <typename T>
	bool operator()(const T* const w_q_i, const T* const w_t_i, const T* w_t_p0, const T* w_t_p1, T* residuals) const
	{
		T i_q_w[4];
		QuaternionInverse(w_q_i, i_q_w);
		//根据相机位姿和地图线两端点位置
		T w_t_ip0[3];
		w_t_ip0[0] = w_t_p0[0] - w_t_i[0];
		w_t_ip0[1] = w_t_p0[1] - w_t_i[1];
		w_t_ip0[2] = w_t_p0[2] - w_t_i[2];

		T i_t_ip0[3];
		ceres::QuaternionRotatePoint(i_q_w, w_t_ip0, i_t_ip0);//计算地图线端点在相机坐标系下的位置

		T w_t_ip1[3];
		w_t_ip1[0] = w_t_p1[0] - w_t_i[0];
		w_t_ip1[1] = w_t_p1[1] - w_t_i[1];
		w_t_ip1[2] = w_t_p1[2] - w_t_i[2];

		T i_t_ip1[3];
		ceres::QuaternionRotatePoint(i_q_w, w_t_ip1, i_t_ip1);//计算地图线端点在相机坐标系下的位置

		T Dxlj = i_t_ip1[0] - i_t_ip0[0];//计算地图线的方向向量
		T Dylj = i_t_ip1[1] - i_t_ip0[1];
		T Dzlj = i_t_ip1[2] - i_t_ip0[2];

		T Dxli = T(i_t_p_x) - i_t_ip0[0];//计算地图线端点到相机坐标系下点的向量
		T Dyli = T(i_t_p_y) - i_t_ip0[1];
		T Dzli = T(i_t_p_z) - i_t_ip0[2];

		T a = Dylj * Dzli - Dzlj * Dyli;//计算地图线端点到相机坐标系下点的向量与地图线的方向向量的叉乘，得到垂直于地图线的向量
		T b = Dzlj * Dxli - Dxlj * Dzli;
		T c = Dxlj * Dyli - Dylj * Dxli;

		T Dvljxvli = pow((a*a + b * b + c * c), 0.5);//计算垂直于地图线的向量的模
		T Dvlj = pow(Dxlj*Dxlj + Dylj * Dylj + Dzlj * Dzlj, 0.5);//计算地图线的长度
		T vvar = pow((T(tx_var)*T(tx_var) + T(ty_var)*T(ty_var) + T(tz_var)*T(tz_var)), 0.5);
		residuals[0] = Dvljxvli/Dvlj  / vvar;//计算地图线端点到相机坐标系下点的向量与地图线的方向向量的叉乘的模与地图线的长度的比值
		//T vvar = pow((T(tx_var)*T(tx_var) + T(ty_var)*T(ty_var) + T(tz_var)*T(tz_var)), 0.5);
		//residuals[0] = Dvljxvli/Dvlj;
		//T xxx = (i_t_ip1[0] - i_t_ip0[0]) / (i_t_ip1[1] - i_t_ip0[1])*(T(i_t_p_y) - i_t_ip0[1]);

		////T yyy = (i_t_ip1[1] - i_t_ip0[1]) / (i_t_ip1[0] - i_t_ip0[0])*(T(i_t_p_x)  -i_t_ip0[0]);
		//residuals[0] = xxx - T(i_t_p_x);

		////residuals[0] = (i_t_ip[0] - T(i_t_p_x)) / T(tx_var);
		////residuals[1] = (i_t_ip[1] - T(i_t_p_y)) / T(ty_var);
		////residuals[2] = (i_t_ip[2] - T(i_t_p_z)) / T(tz_var);

		return true;
	}

	static ceres::CostFunction* Create(const double i_t_p_x, const double i_t_p_y, const double i_t_p_z,
		const double tx_var, const double ty_var, const double tz_var)
	{
		return (new ceres::AutoDiffCostFunction<
			MapLineToFramePointFactor, 1, 4, 3, 3, 3>(
				new MapLineToFramePointFactor(i_t_p_x, i_t_p_y, i_t_p_z,
					tx_var, ty_var, tz_var)));
	}
};

//
//struct MapPointToFramePointFactor
//{
//	MapPointToFramePointFactor(double i_t_p_x, double i_t_p_y, double i_t_p_z,
//		double tx_var, double ty_var, double tz_var)
//		: i_t_p_x(i_t_p_x), i_t_p_y(i_t_p_y), i_t_p_z(i_t_p_z),
//		tx_var(tx_var), ty_var(ty_var), tz_var(tz_var) {}
//
//	double i_t_p_x = 0, i_t_p_y = 0, i_t_p_z = 0;
//	double tx_var = 0, ty_var = 0, tz_var = 0;
//
//	template <typename T>
//	bool operator()(const T* const w_q_i, const T* const w_t_i, const T* w_t_p, T* residuals) const
//	{
//		T w_t_ip[3];
//		w_t_ip[0] = w_t_p[0] - w_t_i[0];
//		w_t_ip[1] = w_t_p[1] - w_t_i[1];
//		w_t_ip[2] = w_t_p[2] - w_t_i[2];
//		T i_q_w[4];
//		QuaternionInverse(w_q_i, i_q_w);
//
//		T i_t_ip[3];
//		ceres::QuaternionRotatePoint(i_q_w, w_t_ip, i_t_ip);
//
//		residuals[0] = (i_t_ip[0] - T(i_t_p_x)) / T(tx_var);
//		residuals[1] = (i_t_ip[1] - T(i_t_p_y)) / T(ty_var);
//		residuals[2] = (i_t_ip[2] - T(i_t_p_z)) / T(tz_var);
//
//		return true;
//
//	}
//	static ceres::CostFunction* Create(const double i_t_p_x, const double i_t_p_y, const double i_t_p_z,
//		const double tx_var, const double ty_var, const double tz_var)
//	{
//		return (new ceres::AutoDiffCostFunction<
//			MapPointToFramePointFactor, 3, 4, 3, 3>(
//				new MapPointToFramePointFactor(i_t_p_x, i_t_p_y, i_t_p_z,
//					tx_var, ty_var, tz_var)));
//	}
//};

struct RelativeRTVarError
{
	/// @brief 
	/// @param t_x 平移向量
	/// @param t_y 
	/// @param t_z 
	/// @param q_w 四元数
	/// @param q_x 
	/// @param q_y 
	/// @param q_z 
	/// @param tx_var 位置的方差
	/// @param ty_var 
	/// @param tz_var 
	/// @param qx_var 旋转的方差
	/// @param qy_var 
	/// @param qz_var 
	RelativeRTVarError(double t_x, double t_y, double t_z,
		double q_w, double q_x, double q_y, double q_z,
		double tx_var, double ty_var, double tz_var,
		double qx_var, double qy_var, double qz_var)
		:t_x(t_x), t_y(t_y), t_z(t_z),
		q_w(q_w), q_x(q_x), q_y(q_y), q_z(q_z),
		tx_var(tx_var), ty_var(ty_var), tz_var(tz_var),
		qx_var(qx_var), qy_var(qy_var), qz_var(qz_var) {}

	template <typename T>
	bool operator()(const T* const w_q_i, const T* ti, const T* w_q_j, const T* tj, T* residuals) const
	{
		//类似于RelativeRTError，但在计算误差时考虑了位置和旋转的方差
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		T i_q_w[4];
		QuaternionInverse(w_q_i, i_q_w);

		T t_i_ij[3];
		ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);

		residuals[0] = (t_i_ij[0] - T(t_x)) / T(tx_var);
		residuals[1] = (t_i_ij[1] - T(t_y)) / T(ty_var);
		residuals[2] = (t_i_ij[2] - T(t_z)) / T(tz_var);

		T relative_q[4];
		relative_q[0] = T(q_w);
		relative_q[1] = T(q_x);
		relative_q[2] = T(q_y);
		relative_q[3] = T(q_z);

		T q_i_j[4];
		ceres::QuaternionProduct(i_q_w, w_q_j, q_i_j);

		T relative_q_inv[4];
		QuaternionInverse(relative_q, relative_q_inv);

		T error_q[4];
		ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);

		residuals[3] = T(2) * error_q[1] / T(qx_var);
		residuals[4] = T(2) * error_q[2] / T(qy_var);
		residuals[5] = T(2) * error_q[3] / T(qz_var);

		return true;
	}

	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
		const double q_w, const double q_x, const double q_y, const double q_z,
		const double tx_var, const double ty_var, const double tz_var,
		const double qx_var, const double qy_var, const double qz_var)
	{
		return (new ceres::AutoDiffCostFunction<
			RelativeRTVarError, 6, 4, 3, 4, 3>(
				new RelativeRTVarError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, tx_var, ty_var, tz_var, qx_var, qy_var, qz_var)));
	}

	double t_x = 0, t_y = 0, t_z = 0, t_norm = 0;
	double q_w = 0, q_x = 0, q_y = 0, q_z = 0;
	double tx_var = 1, ty_var = 1, tz_var = 1;
	double qx_var = 1, qy_var = 1, qz_var = 1;

};

/**
*@brief PoseError Structure for constructing pose error factor
*/
struct PoseVarError
{
	/// @brief 构建姿态误差因子，考虑位置和旋转的方差
	/// @param t_x 平移向量
	/// @param t_y 
	/// @param t_z 
	/// @param q_w 四元数
	/// @param q_x 
	/// @param q_y 
	/// @param q_z 
	/// @param tx_var 位置的方差
	/// @param ty_var 
	/// @param tz_var 
	/// @param qx_var 旋转的方差
	/// @param qy_var 
	/// @param qz_var 
	PoseVarError(double t_x, double t_y, double t_z,
		double q_w, double q_x, double q_y, double q_z,
		double tx_var, double ty_var, double tz_var,
		double qx_var, double qy_var, double qz_var)
		:t_x(t_x), t_y(t_y), t_z(t_z),
		q_w(q_w), q_x(q_x), q_y(q_y), q_z(q_z),
		tx_var(tx_var), ty_var(ty_var), tz_var(tz_var),
		qx_var(qx_var), qy_var(qy_var), qz_var(qz_var) {}

	template <typename T>
	bool operator()(const T* const w_q_i, const T* ti, T* residuals) const
	{
		//类似于RelativeRTError，但是用于单个位姿的误差计算，考虑了位置和旋转的方差。
		residuals[0] = (ti[0] - T(t_x)) / T(tx_var);
		residuals[1] = (ti[1] - T(t_y)) / T(ty_var);
		residuals[2] = (ti[2] - T(t_z)) / T(tz_var);

		T obser_q[4];
		obser_q[0] = T(q_w);
		obser_q[1] = T(q_x);
		obser_q[2] = T(q_y);
		obser_q[3] = T(q_z);

		T q_i[4];
		q_i[0] = T(w_q_i[0]);
		q_i[1] = T(w_q_i[1]);
		q_i[2] = T(w_q_i[2]);
		q_i[3] = T(w_q_i[3]);

		T obser_q_inv[4];
		QuaternionInverse(obser_q, obser_q_inv);

		T error_q[4];
		ceres::QuaternionProduct(obser_q_inv, q_i, error_q);

		residuals[3] = T(2) * error_q[1] / T(qx_var);
		residuals[4] = T(2) * error_q[2] / T(qy_var);
		residuals[5] = T(2) * error_q[3] / T(qz_var);

		return true;
	}

	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
		const double q_w, const double q_x, const double q_y, const double q_z,
		const double tx_var, const double ty_var, const double tz_var,
		const double qx_var, const double qy_var, const double qz_var)
	{
		return (new ceres::AutoDiffCostFunction<
			PoseVarError, 6, 4, 3>(
				new PoseVarError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, tx_var, ty_var, tz_var, qx_var, qy_var, qz_var)));
	}

	double t_x = 0, t_y = 0, t_z = 0, t_norm = 0;
	double q_w = 0, q_x = 0, q_y = 0, q_z = 0;
	double tx_var = 1, ty_var = 1, tz_var = 1;
	double qx_var = 1, qy_var = 1, qz_var = 1;
};

struct PoseLeverVarError
{
	/// @brief 构建带杠杆的姿态误差因子，考虑位置和旋转的方差
	/// @param t_x 平移向量
	/// @param t_y 
	/// @param t_z 
	/// @param q_w 四元数
	/// @param q_x 
	/// @param q_y 
	/// @param q_z 
	/// @param tx_var 位置的方差
	/// @param ty_var 
	/// @param tz_var 
	/// @param qx_var 旋转的方差
	/// @param qy_var 
	/// @param qz_var 
	/// @param lx 杠杆向量
	/// @param ly 
	/// @param lz 
	PoseLeverVarError(double t_x, double t_y, double t_z,
		double q_w, double q_x, double q_y, double q_z,
		double tx_var, double ty_var, double tz_var,
		double qx_var, double qy_var, double qz_var,
		double lx, double ly, double lz)
		:t_x(t_x), t_y(t_y), t_z(t_z),
		q_w(q_w), q_x(q_x), q_y(q_y), q_z(q_z),
		tx_var(tx_var), ty_var(ty_var), tz_var(tz_var),
		qx_var(qx_var), qy_var(qy_var), qz_var(qz_var),
		lx(lx), ly(ly), lz(lz) {}

	template <typename T>
	bool operator()(const T* const w_q_i, const T* ti, T* residuals) const
	{
		//类似于PoseVarError，但在计算误差时考虑了杠杆向量。
		T obser_q[4];
		obser_q[0] = T(q_w);
		obser_q[1] = T(q_x);
		obser_q[2] = T(q_y);
		obser_q[3] = T(q_z);

		T q_i[4];
		q_i[0] = T(w_q_i[0]);
		q_i[1] = T(w_q_i[1]);
		q_i[2] = T(w_q_i[2]);
		q_i[3] = T(w_q_i[3]);

		T itig[3];
		itig[0] = (T)lx;
		itig[1] = (T)ly;
		itig[2] = (T)lz;

		T wtig[3];
		ceres::QuaternionRotatePoint(q_i, itig, wtig);

		residuals[0] = (ti[0] + wtig[0] - T(t_x)) / T(tx_var);
		residuals[1] = (ti[1] + wtig[1] - T(t_y)) / T(ty_var);
		residuals[2] = (ti[2] + wtig[2] - T(t_z)) / T(tz_var);


		T obser_q_inv[4];
		QuaternionInverse(obser_q, obser_q_inv);

		T error_q[4];
		ceres::QuaternionProduct(obser_q_inv, q_i, error_q);

		residuals[3] = T(2) * error_q[1] / T(qx_var);
		residuals[4] = T(2) * error_q[2] / T(qy_var);
		residuals[5] = T(2) * error_q[3] / T(qz_var);

		return true;
	}

	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
		const double q_w, const double q_x, const double q_y, const double q_z,
		const double tx_var, const double ty_var, const double tz_var,
		const double qx_var, const double qy_var, const double qz_var,
		const double lx, const double ly, const double lz)
	{
		return (new ceres::AutoDiffCostFunction<
			PoseLeverVarError, 6, 4, 3>(
				new PoseLeverVarError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, tx_var, ty_var, tz_var, qx_var, qy_var, qz_var, lx, ly, lz)));
	}

	double t_x = 0, t_y = 0, t_z = 0, t_norm = 0;
	double q_w = 0, q_x = 0, q_y = 0, q_z = 0;
	double tx_var = 1, ty_var = 1, tz_var = 1;
	double qx_var = 1, qy_var = 1, qz_var = 1;
	double lx = 0.0, ly = 0.0, lz = 0.0;
};


/**
*@brief PoseError Structure for constructing pose error factor
*/
struct PoseError
{
	/// @brief 构建姿态误差因子
	/// @param t_x 平移向量
	/// @param t_y 
	/// @param t_z 
	/// @param q_w 四元数
	/// @param q_x 
	/// @param q_y 
	/// @param q_z 
	/// @param t_var 位置的方差
	/// @param q_var 旋转的方差
	PoseError(double t_x, double t_y, double t_z,
		double q_w, double q_x, double q_y, double q_z,
		double t_var, double q_var)
		:t_x(t_x), t_y(t_y), t_z(t_z),
		q_w(q_w), q_x(q_x), q_y(q_y), q_z(q_z),
		t_var(t_var), q_var(q_var) {}

	template <typename T>
	bool operator()(const T* const w_q_i, const T* ti, T* residuals) const
	{
		residuals[0] = (ti[0] - T(t_x)) / T(t_var);
		residuals[1] = (ti[1] - T(t_y)) / T(t_var);
		residuals[2] = (ti[2] - T(t_z)) / T(t_var);

		T obser_q[4];
		obser_q[0] = T(q_w);
		obser_q[1] = T(q_x);
		obser_q[2] = T(q_y);
		obser_q[3] = T(q_z);

		T q_i[4];
		q_i[0] = T(w_q_i[0]);
		q_i[1] = T(w_q_i[1]);
		q_i[2] = T(w_q_i[2]);
		q_i[3] = T(w_q_i[3]);

		T obser_q_inv[4];
		QuaternionInverse(obser_q, obser_q_inv);

		T error_q[4];
		ceres::QuaternionProduct(obser_q_inv, q_i, error_q);//计算误差四元数

		residuals[3] = T(2) * error_q[1] / T(q_var);
		residuals[4] = T(2) * error_q[2] / T(q_var);
		residuals[5] = T(2) * error_q[3] / T(q_var);

		return true;
	}

	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
		const double q_w, const double q_x, const double q_y, const double q_z,
		const double t_var, const double q_var)
	{
		return (new ceres::AutoDiffCostFunction<
			PoseError, 6, 4, 3>(
				new PoseError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, t_var, q_var)));
	}

	double t_x = 0, t_y = 0, t_z = 0, t_norm = 0;
	double q_w = 0, q_x = 0, q_y = 0, q_z = 0;
	double t_var = 0, q_var = 0;
};