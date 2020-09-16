#!/usr/bin/env python

from __future__ import division
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from path_generater.srv import UpdateMarker, UpdateMarkerResponse

class PubMarker(object):
    def __init__(self, topic_name, scale, color, type=2):
        self.marker = Marker()
        self.marker.header.frame_id = 'world'
        self.marker.scale.x = scale[0]
        self.marker.scale.y = scale[1]
        self.marker.scale.z = scale[2]
        self.marker.color.r = color[0]
        self.marker.color.g = color[1]
        self.marker.color.b = color[2]
        self.marker.color.a = color[3]
        self.marker.type = type
        self._pub = rospy.Publisher(topic_name, Marker, queue_size=10, latch=True)
        
        
    def publish(self, pose):
        self.marker.action = 2
        self.marker.header.stamp = rospy.Time.now()
        self._pub.publish(self.marker)

        self.marker.action = 0
        self.marker.pose = pose
        self._pub.publish(self.marker)

class MarkerServer(object):
    def __init__(self, topic_ns, service_ns):
        self._markers = dict()
        self._service_name = service_ns+"/update"
        self._topic_name = topic_ns+"/markers"
        self._s = rospy.Service(self._service_name, UpdateMarker, self.request_handler)
        self._pub = rospy.Publisher(self._topic_name, MarkerArray, queue_size=100)
        
    @property
    def service_name(self):
        return self._service_name
    
    @property
    def topic_name(self):
        return self._topic_name

    def clear(self):
        marker = Marker()
        marker.action = Marker.DELETEALL
        self.updateMarker(self, 'DELETE', marker)
        self.applyChange()
    
    def updateMarker(self, name, marker):
        self._markers[name] = marker
    
    def applyChange(self):
        self._pub.publish(self._markers.values())
        for name, marker in self._markers.items():
            if marker.action == Marker.DELETE:
                del self._markers[name]
            elif marker.action == Marker.DELETEALL:
                self._markers = dict()

    def request_handler(self, req):
        req.marker.ns = req.name
        self.updateMarker(req.name, req.marker)
        self.applyChange()
        return UpdateMarkerResponse(success=True, num=len(self._markers.values()))       

def pointMarker(scale, color):
    def makeMarker(posestamped, action):
        marker = Marker()
        marker.header = posestamped.header
        marker.scale.x=scale[0]
        marker.scale.y=scale[1]
        marker.scale.z=scale[2]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.type = Marker.SPHERE
        marker.action = action
        marker.pose = posestamped.pose
        return marker
       
    return makeMarker

if __name__ == "__main__":
    rospy.init_node('test_marker')
    markerServer = MarkerServer(topic_ns='target', service_ns='updateMarker')
    rospy.loginfo("Marker server starts at %s and publish %s", markerServer.service_name, markerServer.topic_name)
    rospy.spin()

      
