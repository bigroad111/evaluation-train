from setuptools import setup, find_packages

NAME = "valuate"
PACKAGES = [NAME] + ["%s.%s" % (NAME, i) for i in find_packages(NAME)]

setup(
    name=NAME,
    version='2.1.11',
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
