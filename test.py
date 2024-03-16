import pkg_resources
import sys

def check_installed_packages():
    installed_packages = [pkg.key for pkg in pkg_resources.working_set]
    return installed_packages

def check_installed_modules():
    installed_modules = sys.modules.keys()
    return [module for module in installed_modules if 'llava' in module]

if __name__ == "__main__":
    # Check installed packages
    installed_packages = check_installed_packages()
    if 'llava' in installed_packages:
        print("Found 'llava' in installed packages.")

    # Check installed modules
    installed_modules = check_installed_modules()
    if installed_modules:
        print("Found modules with 'llava' in the name:", installed_modules)
