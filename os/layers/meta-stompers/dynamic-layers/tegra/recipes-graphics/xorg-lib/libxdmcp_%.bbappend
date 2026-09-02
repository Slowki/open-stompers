# The ZED SDK links against libX11, which the X11-less distro would not build.
REQUIRED_DISTRO_FEATURES:remove = "x11"
