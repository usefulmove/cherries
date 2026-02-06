from setuptools import setup # ,find_packages
from glob import glob

package_name = 'cherry_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[
        package_name,  
        #find_packages()
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name , glob('resource/*.pt')),
        ('share/' + package_name , glob('resource/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wesley',
    maintainer_email='whavener@sabereng.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'detection_service = cherry_detection.detector_node:main',            
        'detector_debug = cherry_detection.detector_debug:main',       
        ],
    },
)
