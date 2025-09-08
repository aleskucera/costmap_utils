import glob

from setuptools import find_packages
from setuptools import setup

package_name = "costmap_utils"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob.glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="kuceral4",
    maintainer_email="kuceral4@fel.cvut.cz",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "geometric_traversability_node = costmap_utils.geometric_traversability_node:main"
        ],
    },
)
