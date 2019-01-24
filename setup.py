from setuptools import setup, find_packages

setup(name='mabalgs',
      version='0.3',
      description='Multi-armed bandit algorithms',
      url='https://github.com/alison-carrera/mabalgs',
      author='Alison Carrera',
      author_email='alison.carrera2007@gmail.com',
      packages=find_packages(),
      install_requires=['numpy'],
      license='Apache 2.0',
      zip_safe=False)