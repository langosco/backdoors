from setuptools import setup

setup(name='backdoors',  # For pip. E.g. `pip show`, `pip uninstall`
      version='0.0.1',
      author="Lauro Langosco",
      packages=["backdoors"], # For python. E.g. `import python_template`
      install_requires=[
          "jax",
          "flax",
          "numpy",
          "einops",
          "chex",
          ],
      )
