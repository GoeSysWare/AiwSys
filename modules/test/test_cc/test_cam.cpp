#include "usb_cam.h"

using namespace usb_cam;

int main()
{
    UsbCam cam;

    cam.start("/dev/video4",
    usb_cam::UsbCam::IO_METHOD_MMAP,
    usb_cam::UsbCam::PIXEL_FORMAT_YUYV,
    1920,1080,10);

    cam.grab_image();
    return 0;
}