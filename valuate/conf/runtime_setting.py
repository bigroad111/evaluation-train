from sqlalchemy import create_engine
from valuate.conf import global_settings as gl

# 公共配置
ENCODING = gl.ENCODING

# 根据运行环境分别配置
if gl.RUNTIME_ENVIRONMENT == 'PRODUCT':
    ENGINE = create_engine(gl.PRODUCE_VALUATE_ENGINE_INNER_MYSQL_CON, pool_recycle=1, encoding=gl.ENCODING)
elif gl.RUNTIME_ENVIRONMENT == 'LOCAL':
    ENGINE = create_engine(gl.PRODUCE_VALUATE_ENGINE_OUTTER_PYMYSQL, pool_recycle=1, encoding=gl.ENCODING)