cc_library(
    name = "qt_core",
    hdrs = glob(["*"]),
    copts = [
        "-Iinclude",
        "-Iinclude/QtCore",
    ],
    includes = [
        "include",
        "include/QtCore",
    ],
    linkopts = [
        "-Wl,-rpath,/opt/Qt5.5.1/5.5/gcc_64/lib",
        "-lQt5Core",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "qt_widgets",
    hdrs = glob(["*"]),
    copts = [
        "-Iinclude",
        "-Iinclude/QtWidgets",
    ],
    includes = ["include/QtWidgets"],
    linkopts = [
        "-L/opt/Qt5.5.1/5.5/gcc_64/lib",
        "-lQt5Widgets",
    ],
    visibility = ["//visibility:public"],
    deps = [":qt_core"],
)

cc_library(
    name = "qt_gui",
    hdrs = glob(["*"]),
    copts = [
        "-Iinclude",
        "-Iinclude/QtGui",
    ],
    includes = ["include/QtGui"],
    linkopts = [
        "-L/opt/Qt5.5.1/5.5/gcc_64/lib",
        "-lQt5Gui",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":qt_core",
        ":qt_widgets",
    ],
)

cc_library(
    name = "qt_opengl",
    hdrs = glob(["*"]),
    copts = [
        "-Iinclude",
        "-Iinclude/QtCore",
        "-Iinclude/QtWidgets",
        "-Iinclude/QtGui",
        "-Iinclude/QtOpenGL",
    ],
    includes = ["include/QtOpenGL"],
    linkopts = [
        "-L/opt/Qt5.5.1/5.5/gcc_64/lib",
        "-lQt5OpenGL",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":qt_core",
        ":qt_gui",
        ":qt_widgets",
    ],
)

cc_library(
    name = "qt_quick",
    hdrs = glob(["*"]),
    copts = [
        "-Iinclude",
        "-Iinclude/QtCore",
        "-Iinclude/QtWidgets",
        "-Iinclude/QtGui",
        "-Iinclude/QtQuick",
    ],
    includes = ["include/QtQuick"],
    linkopts = [
        "-L/opt/Qt5.5.1/5.5/gcc_64/lib",
        "-lQt5Quick",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":qt_core",
        ":qt_gui",
        ":qt_widgets",
    ],
)

cc_library(
    name = "qt_qml",
    hdrs = glob(["*"]),
    copts = [
        "-Iinclude",
        "-Iinclude/QtCore",
        "-Iinclude/QtWidgets",
        "-Iinclude/QtGui",
        "-Iinclude/QtQml",
    ],
    includes = ["include/QtQml"],
    linkopts = [
        "-L/opt/Qt5.5.1/5.5/gcc_64/lib",
        "-lQt5Qml",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":qt_core",
        ":qt_gui",
        ":qt_widgets",
    ],
)

cc_library(
    name = "qt_media",
    hdrs = glob(["*"]),
    copts = [
        "-Iinclude",
        "-Iinclude/QtCore",
        "-Iinclude/QtWidgets",
        "-Iinclude/QtGui",
        "-Iinclude/QtMultimedia",
        "-Iinclude/QtMultimediaWidgets",
    ],
    includes = [
        "include/QtMultimedia",
        "include/QtMultimediaWidgets",
        ],
    linkopts = [
        "-L/opt/Qt5.5.1/5.5/gcc_64/lib",
        "-lQt5Multimedia",
        "-lQt5MultimediaWidgets",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":qt_core",
        ":qt_gui",
        ":qt_widgets",
    ],
)
cc_library(
    name = "qt_network",
    hdrs = glob(["*"]),
    copts = [
        "-Iinclude",
        "-Iinclude/QtCore",
        "-Iinclude/QtWidgets",
        "-Iinclude/QtGui",
        "-Iinclude/QtNetwork",
    ],
    includes = [
        "include/QtNetwork",
        ],
    linkopts = [
        "-L/opt/Qt5.5.1/5.5/gcc_64/lib",
        "-lQt5Network",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":qt_core",
        ":qt_gui",
        ":qt_widgets",
    ],
)

