#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

# CONFIG_FILE = "dstest1_pgie_config.txt"
CONFIG_FILE = "config_infer_primary_yoloV5.txt"     # TODO
OUTPUT_DIR = "/home/cyclope/iLori/repo/DeepStream_Test/pipeline_test/result/"    # TODO
RUNTIME = 1     # TODO

import time
from timeit import timeit
import numpy as np
import cv2

import argparse
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream-6.2/sources/deepstream_python_apps/apps')
sys.path.append('.')

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call

import pyds

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_OBJECT = 1
PGIE_CLASS_ID_PERSON = 2


def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_OBJECT:0,
    }
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.class_id in obj_counter:
                obj_counter[obj_meta.class_id] += 1

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 8
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 0.4)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.4)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)


def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <media file or uri>\n" % args[0])
        sys.exit(1)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    # QTDemux for demuxing different type of input streams
    qtdemux = Gst.ElementFactory.make("qtdemux", "qtdemux")
    if not qtdemux:
        sys.stderr.write(" Unable to create QTDemux \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # ------------------

    # Create Bin for source file
    bin_name="source-bin"
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")

    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",args[1])
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")

    queue1=Gst.ElementFactory.make("queue","queue1")

    # ------------------

    # Since the data format in the input file is elementary h264 stream,
    # we need a h264parser
    print("Creating H264Parser \n")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        sys.stderr.write(" Unable to create h264 parser \n")
    
    # Use nvdec_h264 for hardware accelerated decode on GPU
    print("Creating Decoder \n")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")
    
    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    
    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    
    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    # nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    # if not nvvidconv_postosd:
    #     sys.stderr.write(" Unable to create nvvidconv_postosd \n")
    
    # # Create a caps filter
    # caps = Gst.ElementFactory.make("capsfilter", "filter")
    # caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
    
    # # Make the encoder
    # if codec == "H264":
    #     encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    #     print("Creating H264 Encoder")
    # elif codec == "H265":
    #     encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
    #     print("Creating H265 Encoder")
    # if not encoder:
    #     sys.stderr.write(" Unable to create encoder")
    # encoder.set_property('bitrate', bitrate)
    # if is_aarch64():
    #     encoder.set_property('preset-level', 1)
    #     encoder.set_property('insert-sps-pps', 1)
    #     #encoder.set_property('bufapi-version', 1)
    
    # # Make the payload-encode video into RTP packets
    # if codec == "H264":
    #     rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
    #     print("Creating H264 rtppay")
    # elif codec == "H265":
    #     rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
    #     print("Creating H265 rtppay")
    # if not rtppay:
    #     sys.stderr.write(" Unable to create rtppay")

    # Finally encode and save the osd output
    queue = Gst.ElementFactory.make("queue", "queue")
    if not queue:
        sys.stderr.write(" Unable to create Queue \n")

    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")

    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")

    caps = Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)

    print("Creating Encoder \n ")
    encoder = Gst.ElementFactory.make("avenc_mpeg4", "encoder")
    if not encoder:
        sys.stderr.write(" Unable to create Encoder \n")

    encoder.set_property("bitrate", 2000000)

    print("Creating CodeParser \n ")
    codeparser = Gst.ElementFactory.make("mpeg4videoparse", "mpeg4-parser")
    if not codeparser:
        sys.stderr.write(" Unable to create CodeParser \n")

    print("Creating Container \n ")
    container = Gst.ElementFactory.make("qtmux", "qtmux")
    if not container:
        sys.stderr.write(" Unable to create Container \n")

    # Make the sink
    sink = Gst.ElementFactory.make("filesink", "filesink")
    if not sink:
        sys.stderr.write(" Unable to create Sink")

    # Output video name
    video_name = args[1].split("/")[-1]
    output_video_name = OUTPUT_DIR + video_name

    sink.set_property("location", output_video_name)
    sink.set_property("sync", 0)
    sink.set_property("async", 0)
    
    # Make the UDP sink
    # updsink_port_num = 5400
    # sink = Gst.ElementFactory.make("udpsink", "udpsink")
    # if not sink:
    #     sys.stderr.write(" Unable to create udpsink")
    
    # sink.set_property('host', '224.224.255.255')
    # sink.set_property('port', updsink_port_num)
    # sink.set_property('async', False)
    # sink.set_property('sync', 1)
    
    # print("Playing file %s " %stream_path)
    # source.set_property('location', stream_path)
    # streammux.set_property('width', 1920)
    # streammux.set_property('height', 1080)
    # streammux.set_property('batch-size', 1)
    # streammux.set_property('batched-push-timeout', 4000000)

    print("Playing file %s " % args[1])
    source.set_property("location", args[1])
    streammux.set_property("width", WIDTH)
    streammux.set_property("height", HEIGHT)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)
    
    pgie.set_property('config-file-path', CONFIG_FILE)
    
    print("Adding elements to Pipeline \n")
    # --pipeline.add(source)
    # ---pipeline.add(qtdemux)  # ---
    # --pipeline.add(h264parser)
    # --pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(nbin)  # --
    pipeline.add(queue1) # --
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    # pipeline.add(nvvidconv_postosd)
    # pipeline.add(caps)
    # pipeline.add(encoder)
    # pipeline.add(rtppay)
    # pipeline.add(sink)
    pipeline.add(queue)
    pipeline.add(nvvidconv2)
    pipeline.add(capsfilter)
    pipeline.add(encoder)
    pipeline.add(codeparser)
    pipeline.add(container)
    pipeline.add(sink)

    # Link the elements together:
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> nvvidconv_postosd -> 
    # caps -> encoder -> rtppay -> udpsink

    # --
    # we link the elements together
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> video-renderer
    
    print("Linking elements in the Pipeline \n")
    # --source.link(h264parser)
    # source.link(qtdemux)  # ---
    # qtdemux.link(h264parser)  # ---
    # --h264parser.link(decoder)

    # --
    # sinkpad = streammux.get_request_pad("sink_0")
    # if not sinkpad:
    #     sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
    # srcpad = decoder.get_static_pad("src")
    # if not srcpad:
    #     sys.stderr.write(" Unable to get source pad of decoder \n")
    
    # --srcpad.link(sinkpad)

    sinkpad = streammux.get_request_pad("sink_0") 
    if not sinkpad:
        sys.stderr.write("Unable to create sink pad bin \n")

    srcpad = nbin.get_static_pad("src")
    if not srcpad:
        sys.stderr.write("Unable to create src pad bin \n")

    srcpad.link(sinkpad)

    # --streammux.link(pgie)
    # --pgie.link(nvvidconv)
    streammux.link(queue1)  # --
    queue1.link(pgie)   # --
    pgie.link(nvvidconv)  # --
    nvvidconv.link(nvosd)
    # nvosd.link(nvvidconv_postosd)
    # nvvidconv_postosd.link(caps)
    # caps.link(encoder)
    # encoder.link(rtppay)
    # rtppay.link(sink)
    nvosd.link(queue)
    queue.link(nvvidconv2)
    nvvidconv2.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(codeparser)
    codeparser.link(container)
    container.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    # Start streaming
    # rtsp_port_num = 8554
    
    # server = GstRtspServer.RTSPServer.new()
    # server.props.service = "%d" % rtsp_port_num
    # server.attach(None)
    
    # factory = GstRtspServer.RTSPMediaFactory.new()
    # factory.set_launch( "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (updsink_port_num, codec))
    # factory.set_shared(True)
    # server.get_mount_points().add_factory("/ds-test", factory)
    
    # print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)

    #-- Add a probe on the primary-infer source pad to get inference output tensors
    # pgiesrcpad = pgie.get_static_pad("src")
    # if not pgiesrcpad:
    #     sys.stderr.write(" Unable to get src pad of primary infer \n")

    # pgiesrcpad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)
    
    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(description='Test pipeline: decode -> inference -> postprocess -> encode -> save')

    print("NOTE: file path should add 'file://' in front of the path")

    # Check input arguments
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return 0


if __name__ == '__main__':
    parse_args()

    # time = timeit('main(sys.argv)', number = 50)
    # print('!!!Total execution time: ', time)

    cap = cv2.VideoCapture(sys.argv[1])
    WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    times = RUNTIME
    time_list = []
    for i in range(times):
        start = time.time()
        main(sys.argv)
        end = time.time()
        execution = end - start

        print("  !!!Current run: ", i)
        print("  !!!Total execution time: ", execution)
        time_list.append(execution)

    print("!!!Time List:")
    print(time_list)

    average_time = np.mean(time_list)
    print("!!!Average time: ", average_time)

    std_time = np.std(time_list)
    print("!!!Variance: ", std_time)

    sys.exit(0)
