from setuptools import setup, find_packages

# "N.N": 版本号里唯一必备的部分，两个"N"分别代表了主版本和副版本号，绝大多数现存的工程里都包含该部分。
# "[.N]": 次要版本号，可以有零或多个。
# "{a|b|c|rc}": 阶段代号，a, b, c, rc分别代表alpha, beta, candidate 和 release candidate
# 例如:'3.2.3.2c5.4.12' （3.2.3.2 的 candidate 阶段的 5.4.12 版本 正式版
NAME = "valuate"
PACKAGES = [NAME] + ["%s.%s" % (NAME, i) for i in find_packages(NAME)]

setup(
    name=NAME,
    version='2.1.6',
    author='DJ Leo',
    author_email='m18349125880@gmail.com',
    description='Used car valuation module',
    packages=PACKAGES,

    install_requires=[
        'scipy==0.19.1',
        'numpy==1.13.1',
        'pandas==0.20.2',
        'setuptools==20.7.0',
        'scikit_learn==0.19.0',
        'SQLAlchemy==1.1.11',
        'mysql-connector-python==8.0.5',
        'PyMySQL==0.7.2',
    ],

    include_package_data=True
)
