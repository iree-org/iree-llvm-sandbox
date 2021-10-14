# Debug utils
def inspect(obj):
  print(obj)
  print([name for name in dir(obj) if not name.startswith('__')])


def inspect_all(obj):
  inspect(obj)
  print([name for name in dir(obj) if name.startswith('__')])
