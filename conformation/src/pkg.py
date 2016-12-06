import json
import requests
from distutils.version import StrictVersion

def versions(package_name):
    url = "https://pypi.python.org/pypi/{}/json".format(package_name)
    data = requests.get(url).json()
    return sorted(list(data["releases"].keys()), key=StrictVersion, reverse=True)

