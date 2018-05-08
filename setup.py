from setuptools import setup, find_packages

setup(
    name = 'PySFD',
    version = '1.0.0.dev0',
    url = 'https://github.com/markovmodel/PySFD.git',
    author = 'Sebastian Stolzenberg',
    author_email = 'ss629@cornell.edu',
    description = 'PySFD - Significant Feature Differences Analyzer for Python',
    packages = find_packages(),    
    #install_requires = [ 'biopandas', 'jupyter', 'numpy', 'matplotlib', 'mdtraj', 'pandas', 'pathos' ],
    zip_safe = False,
    include_package_data = True
)
