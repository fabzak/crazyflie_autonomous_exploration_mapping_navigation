from glob import glob

from setuptools import setup

package_name = 'cf_explore'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/layer_explore.launch.py',
            'launch/real_base.launch.py',
            'launch/layer_explore_real.launch.py',
            'launch/cf_auto.launch.py',
            'launch/cf_auto_real.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/cf_auto.yaml',
            'config/cf_auto.rviz',
            'config/crazyflies_real.yaml',
            'config/layer_explore_real.yaml',
            'config/layer_explore_real.rviz',
            'config/cf_auto_real.yaml',
            'config/real_safety.yaml',
        ]),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools','numpy'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@example.com',
    description='Crazyflie layer mapping and saved-map autonomous flight (ROS 2).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'layer_explore = cf_explore.layer_explore:main',
            'range_scan_merger = cf_explore.range_scan_merger:main',
            'cf_auto = cf_explore.cf_auto:main',
            'cf_auto_layer_visualizer = '
            'cf_explore.cf_auto_layer_visualizer:main',
            'cf_auto_planar_frame = cf_explore.cf_auto_planar_frame:main',
            'real_sensor_adapter = cf_explore.real_sensor_adapter:main',
            'real_control_adapter = cf_explore.real_control_adapter:main',
            'real_safety_watchdog = cf_explore.real_safety_watchdog:main',
            'real_operator_control = '
            'cf_explore.real_operator_control:main',
            'real_body_frame = cf_explore.real_body_frame:main',
        ],
    },
)
