from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='experimenting image retrieval on gcloud ml-engine',
      author='Anirudha Vishvakarma',
      author_email='anirudhav@gofynd.com',
      license='MIT',
      install_requires=[
          # 'tensorflow==1.10',
          'hyperdash',
          'pillow',
          # 'keras==2.0.4',
          'h5py'
      ],
      zip_safe=False)
