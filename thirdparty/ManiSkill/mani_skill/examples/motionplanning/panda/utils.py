import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import trimesh
from transforms3d import quaternions
from mani_skill.utils.structs import Actor
from mani_skill.utils import common
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh


def get_actor_obb(actor: Actor, to_world_frame=True, vis=False):
    mesh = get_component_mesh(
        actor._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent),
        to_world_frame=to_world_frame,
    )
    assert mesh is not None, "can not get actor mesh for {}".format(actor)

    obb: trimesh.primitives.Box = mesh.bounding_box_oriented

    if vis:
        obb.visual.vertex_colors = (255, 0, 0, 10)
        trimesh.Scene([mesh, obb]).show()

    return obb


def compute_grasp_info_by_obb(
    obb: trimesh.primitives.Box,
    approaching=(0, 0, -1),
    target_closing=None,
    depth=0.0,
    ortho=True,
):
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.

    Args:
        obb: oriented bounding box to grasp
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # NOTE(jigu): DO NOT USE `x.extents`, which is inconsistent with `x.primitive.transform`!
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)

    # Assume normalized
    approaching = np.array(approaching)

    # Find the axis closest to approaching vector
    angles = approaching @ T[:3, :3]  # [3]
    inds0 = np.argsort(np.abs(angles))
    ind0 = inds0[-1]

    # Find the shorter axis as closing vector
    inds1 = np.argsort(extents[inds0[0:-1]])
    ind1 = inds0[0:-1][inds1[0]]
    ind2 = inds0[0:-1][inds1[1]]

    # If sizes are close, choose the one closest to the target closing
    if target_closing is not None and 0.99 < (extents[ind1] / extents[ind2]) < 1.01:
        vec1 = T[:3, ind1]
        vec2 = T[:3, ind2]
        if np.abs(target_closing @ vec1) < np.abs(target_closing @ vec2):
            ind1 = inds0[0:-1][inds1[1]]
            ind2 = inds0[0:-1][inds1[0]]
    closing = T[:3, ind1]

    # Flip if far from target
    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    # Reorder extents
    extents = extents[[ind0, ind1, ind2]]

    # Find the origin on the surface
    center = T[:3, 3].copy()
    half_size = extents[0] * 0.5
    center = center + approaching * (-half_size + min(depth, half_size))

    if ortho:
        closing = closing - (approaching @ closing) * approaching
        closing = common.np_normalize_vector(closing)

    grasp_info = dict(
        approaching=approaching, closing=closing, center=center, extents=extents
    )
    return grasp_info


def compute_grasp_info_of_faucet(faucet_pose, approaching, depth: float = 0.0, ortho: bool = True) -> dict:
    """
    根据水龙头的基坐标（faucet_pose），计算抓取信息，逻辑如下：
      - 固定抓取点在水龙头基坐标系中的 (-0.15, 0, 0.2) 处，
        并通过 faucet_pose 转换到全局坐标，若 depth 非零，则沿 approaching 方向加上偏移。
      - 定义抓取时的 approaching 方向（直接使用传入值）
      - 定义 closing 方向为 faucet_pose 的旋转作用下固定的局部向量 [-1, 0, 0]
      - extents 返回一个空值（或零向量），因为此方法不依赖网格几何信息

    参数：
      faucet_pose: 表示水龙头基坐标的 sapien.Pose 对象（原接口中 mesh_list 的位置，现在用来传入 faucet_pose）
      approaching: 接近方向，numpy 数组
      depth: 沿 approaching 方向的偏移量，默认 0.0
      ortho: 是否保持 closing 与 approaching 正交，默认为 True（本例中不做额外处理）
      
    返回：
      一个字典，包含键：
        "approaching": 与传入 approaching 相同
        "closing": 由 faucet_pose 的旋转作用下的局部 closing 向量 [-1,0,0] 得到的全局 closing 方向
        "center": 计算得到的抓取点全局坐标（即 faucet_pose 转换 (-0.15, 0, 0.2)，再加上 depth 偏移）
        "extents": 默认返回一个零向量（不使用网格几何信息）
    """
    import numpy as np
    # 固定抓取点在水龙头局部坐标系中
    local_center = np.array([-0.11, 0.05, 0.125])
    # 将局部抓取点转换为全局坐标：即 faucet_pose 作用于 local_center
    global_center = (faucet_pose * sapien.Pose(p=local_center)).p
    # 如果 depth 非零，则在 approaching 方向上加上偏移
    if depth != 0.0:
        global_center = global_center + approaching * depth

    global_closing = np.array([0, -1, 0.0])

    
    extents = np.zeros(3)

    grasp_info = {
        "approaching": approaching,
        "closing": global_closing,
        "center": global_center,
        "extents": extents,
    }
    return grasp_info


def compute_grasp_info_of_laptop(laptop_pose, approaching, depth: float = 0.0, ortho: bool = True) -> dict:
    """
    根据水龙头的基坐标（faucet_pose），计算抓取信息，逻辑如下：
      - 固定抓取点在水龙头基坐标系中的 (-0.15, 0, 0.2) 处，
        并通过 faucet_pose 转换到全局坐标，若 depth 非零，则沿 approaching 方向加上偏移。
      - 定义抓取时的 approaching 方向（直接使用传入值）
      - 定义 closing 方向为 faucet_pose 的旋转作用下固定的局部向量 [-1, 0, 0]
      - extents 返回一个空值（或零向量），因为此方法不依赖网格几何信息

    参数：
      faucet_pose: 表示水龙头基坐标的 sapien.Pose 对象（原接口中 mesh_list 的位置，现在用来传入 faucet_pose）
      approaching: 接近方向，numpy 数组
      depth: 沿 approaching 方向的偏移量，默认 0.0
      ortho: 是否保持 closing 与 approaching 正交，默认为 True（本例中不做额外处理）
      
    返回：
      一个字典，包含键：
        "approaching": 与传入 approaching 相同
        "closing": 由 faucet_pose 的旋转作用下的局部 closing 向量 [-1,0,0] 得到的全局 closing 方向
        "center": 计算得到的抓取点全局坐标（即 faucet_pose 转换 (-0.15, 0, 0.2)，再加上 depth 偏移）
        "extents": 默认返回一个零向量（不使用网格几何信息）
    """
    import numpy as np
    # 固定抓取点在水龙头局部坐标系中
    local_center = np.array([-0.19, -0.15, 0.04])
    
    # 将局部抓取点转换为全局坐标：即 faucet_pose 作用于 local_center
    global_center = (laptop_pose * sapien.Pose(p=local_center)).p
    # 如果 depth 非零，则在 approaching 方向上加上偏移
    if depth != 0.0:
        global_center = global_center + approaching * depth

    global_closing = np.array([0, -1, 0.0])

    
    extents = np.zeros(3)

    grasp_info = {
        "approaching": approaching,
        "closing": global_closing,
        "center": global_center,
        "extents": extents,
    }
    return grasp_info
