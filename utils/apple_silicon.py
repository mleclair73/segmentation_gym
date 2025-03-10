import platform
is_apple_silicon = platform.system() == 'Darwin' and platform.processor() == 'arm'
