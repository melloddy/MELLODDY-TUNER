import setuptools

exec(open('melloddy_tuner/version.py').read())

setuptools.setup(
      name='melloddy_tuner',
      version=__version__,
      author='Lukas Friedrich',
      author_mail='lukas.friedrich@merckgroup.com',
      description='MELLODDY-melloddy_tuner',
      long_description='Processing of structure and activity data for machine learning models in drug discovery',
      long_description_content_type='text/markdown',
      url='www.melloddy.eu',
      packages=setuptools.find_packages()
)