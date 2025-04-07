python -m mani_skill.trajectory.replay_trajectory \
    --traj-path "/home/chenshan/robotics/dex-retargeting/manitask/data/h5/PlaceSphere-v1/ur5e_allegro_right/origin/episode_10/trajectory.h5" \
    --save_traj \
    --save-video \
    --obs-mode "rgb+depth+segmentation" \
    --use_first_env_state

