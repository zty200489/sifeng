import io
import os
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command

# Meta-data
NAME = 'sifeng'
DESCRIPTION = 'The machine learning quant research framework.'
URL = 'https://github.com/zty200489/sifeng'
EMAIL = 'zty200489@gmail.com'
AUTHOR = 'Tianyuan Zhou'
REQUIRES_PYTHON = '>=3.8.0'

# Required Packages
REQUIRED = [
    "numpy",
    "pandas",
    "duckdb",
    "joblib",
    "requests",
    "tqdm",
    "colorama",
]

# Optional Packages
EXTRAS = {
}

# Global Settings
here = os.path.abspath(os.path.dirname(__file__))
# Load REAMDE.md
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load __version__.py
about = {}
project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
with open(os.path.join(here, project_slug, '__version__.py')) as f:
    exec(f.read(), about)

# Upload Commands
class UploadCommand(Command):
    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass
        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))
        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')
        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')
        sys.exit()


# Setup Process
setup(
    name = NAME,
    version = about['__version__'],
    description = DESCRIPTION,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = AUTHOR,
    author_email = EMAIL,
    python_requires = REQUIRES_PYTHON,
    url = URL,
    packages = find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['sifeng'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires = REQUIRED,
    extras_require = EXTRAS,
    include_package_data = True,
    license = 'GPL',
    classifiers = [
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    # $ setup.py publish support.
    # cmdclass = {
    #     'upload': UploadCommand,
    # },
)
