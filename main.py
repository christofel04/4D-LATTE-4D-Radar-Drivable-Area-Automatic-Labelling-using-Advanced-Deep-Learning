#!/usr/bin/python3

''' A basic GUi to use ImageViewer class to show its functionalities and use cases. '''

from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from actions import ImageViewer

# Package for Drivable Area segmentation
from segment_anything import SamPredictor, sam_model_registry

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import sys, os

from PyQt5.QtCore import QLibraryInfo

import mahotas

# Package for LiDAR point cloud processing for projecting DA Label
import open3d as o3d

import yaml

from scipy.spatial.transform import Rotation as R

#os.environ[ "QT_QPA_PLATFORM" ] = "offscreen"

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

DIR_OF_SEGMENT_ANYTHING_MODEL = "/home/ofel04/Downloads/sam_vit_l_0b3195.pth"

FOLDER_OF_ROOT_KRADAR_DATASET = "/home/ofel04/Downloads/"

gui = uic.loadUiType("main.ui")[0]     # load UI file designed in Qt Designer
VALID_FORMAT = ('.BMP', '.GIF', '.JPG', '.JPEG', '.PNG', '.PBM', '.PGM', '.PPM', '.TIFF', '.XBM')  # Image formats supported by Qt

def getImages(folder):
    ''' Get the names and paths of all the images in a directory. '''
    image_list = []
    if os.path.isdir(folder):
        for file in sorted( os.listdir(folder)):
            if file.upper().endswith(VALID_FORMAT):
                im_path = os.path.join(folder, file)
                image_obj = {'name': file, 'path': im_path }
                image_list.append(image_obj)
    return image_list

class Iwindow(QtWidgets.QMainWindow, gui):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.cntr, self.numImages = -1, -1  # self.cntr have the info of which image is selected/displayed
        self.folder = None

        self.image_viewer = ImageViewer(self.qlabel_image)
        self.drivable_area_image_viewer = ImageViewer(self.drivable_area_qlabel_image , image_position=[0,300])
        self.bev_drivable_area_image_viewer = ImageViewer(self.bev_drivable_area_qlabel_image, image_position=[0,0])
        self.bev_drivable_area_previous_image_q_label_image_viewer = ImageViewer( self.bev_drivable_area_previous_image_q_label_image)
        
        self.is_add_attention_point : bool = False
        self.list_of_attention_points = []
        self.list_of_attention_points_mark_on_image = []
        self.is_delete_attention_point : bool = False  

        self.is_add_drivable_area_in_bev_drivable_area : bool = False 
        self.delete_drivable_area_on_image_start_point = None 

        self.add_bev_drivable_area_start_point = None 
        self.delete_bev_drivable_area_start_point = None 

        self.list_of_attention_points_non_drivable_area = []
        self.list_of_attention_points_non_drivable_area_mark_on_image = []

        # Load camera to LiDAR point cloud transformation matrix from camera clibration parameter

        with open("resources/cam_calib/common/cam_front0.yml") as stream:
            try:
                
                camera_calibration_parameter = dict( yaml.safe_load(stream))
                #print(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)
        camera_intrinsic_parameter = np.zeros((3,3))

        camera_intrinsic_parameter[0][0] = camera_calibration_parameter["fx"]
        camera_intrinsic_parameter[1][1] = camera_calibration_parameter["fy"]
        camera_intrinsic_parameter[0][2] = camera_calibration_parameter["px"]
        camera_intrinsic_parameter[1][2] = camera_calibration_parameter["py"]
        camera_intrinsic_parameter[2][2] = 1

        self.camera_intrinsic_parameter = camera_intrinsic_parameter

        yaw = float( camera_calibration_parameter["yaw_ldr2cam"] )
        pitch = float( camera_calibration_parameter["pitch_ldr2cam"] )
        roll = float( camera_calibration_parameter["roll_ldr2cam"])

        r = R.from_euler("zyx", (yaw, pitch, roll), degrees=True)

        r.as_matrix()

        camera_extrinsic_transformation_matrix = np.zeros( (3,3 ) )

        camera_extrinsic_transformation_matrix[ : 3 , : 3 ] = r.as_matrix()


        camera_extrinsic_transformation_matrix

        self.camera_extrinsic_transformation_matrix = camera_extrinsic_transformation_matrix

        # Load Drivable Area Segmentation Model
        print( "Loading Drivable Area Segmentation for Labelling...")
        drivable_segmentation_model = sam_model_registry["vit_l"](DIR_OF_SEGMENT_ANYTHING_MODEL )
        self.drivable_area_predictor = SamPredictor(drivable_segmentation_model)

        if self.drivable_area_predictor :
            print( "Successfully loading Drivable Area Segmentation Model" )

        self.center = None

        self.__connectEvents()
        self.showMaximized()

        #self._pixmap_item = QtWidgets.QGraphicsPixmapItem()

    def __connectEvents(self):
        self.open_folder.clicked.connect(self.selectDir)
        self.next_im.clicked.connect(self.nextImg)
        self.prev_im.clicked.connect(self.prevImg)
        self.qlist_images.itemClicked.connect(self.item_click)
        # \\self.save_im.clicked.connect(self.saveImg)

        self.add_attention_points.clicked.connect(self.add_attention_point_button_clicked)
        self.add_BEV_DA_Label_from_LiDAR.clicked.connect( self.add_bev_drivable_area_label_button_clicked )
        self.delete_BEV_DA_Label_from_LiDAR.clicked.connect( self.delete_bev_drivable_area_label_button_clicked )
        self.image_viewer.qlabel_image.mousePressEvent= self.add_attention_point
        self.image_viewer.qlabel_image.mouseReleaseEvent= self.mouseReleaseEvent_for_add_attention_point

        self.drivable_area_image_viewer.qlabel_image.mousePressEvent = self.delete_drivable_area_label_MousePressedButton
        self.drivable_area_image_viewer.qlabel_image.mouseReleaseEvent = self.delete_drivable_area_label_MouseReleaseButton

        self.bev_drivable_area_image_viewer.qlabel_image.mousePressEvent = self.add_drivable_area_in_bev_drivable_area_MousePressedButton
        self.bev_drivable_area_image_viewer.qlabel_image.mouseReleaseEvent = self.add_drivable_area_in_bev_drivable_area_MouseReleasedButton

        #self.list_of_attention_point_column.mousePressEvent = self.list_of_attention_point_column_check_if_left_mouse_pressed
        self.list_of_attention_point_column.itemClicked.connect(self.delete_attention_points)
        self.list_of_attention_points_non_drivable_area_column.itemClicked.connect(self.delete_attention_points_non_drivable_area)
        #self.qlist
        self.generate_DA_label.clicked.connect(self.predict_drivable_area )
        #self.fix_holes_DA_label_button.clicked.connect( self.fix_holes_drivable_area_label )
        self.generates_BEV_DA_Label_from_LiDAR.clicked.connect( self.generates_bev_DA_label_projected_to_lidar_pcd )
        self.save_Drivable_Area_label.clicked.connect( self.save_drivable_area_label )

        self.copy_BEV_DA_Label_from_Previous_Frame.clicked.connect( self.copy_bev_drivable_area_label_from_previous_lidar_frame_label )
        self.load_BEV_Drivable_Area_Label.clicked.connect( self.load_bev_drivable_area_label_from_folder )

        self.clear_all_attention_points_button.clicked.connect(self.delete_all_attention_points )
        self.zoom_plus.clicked.connect(self.image_viewer.zoomPlus)
        self.zoom_minus.clicked.connect(self.image_viewer.zoomMinus)
        self.reset_zoom.clicked.connect(self.image_viewer.resetZoom)

        self.toggle_line.toggled.connect(self.action_line)
        self.toggle_rect.toggled.connect(self.action_rect)
        self.toggle_move.toggled.connect(self.action_move)

    def selectDir(self):
        ''' Select a directory, make list of images in it and display the first image in the list. '''
        # open 'select folder' dialog box
        self.folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        if not self.folder:
            QtWidgets.QMessageBox.warning(self, 'No Folder Selected', 'Please select a valid Folder')
            return
        
        # Load camera image timestamp and LiDAR pcd timestamp in K- Radar dataset

        try :

            index_of_Kradar_scene_in_camera_image = int( self.folder.split( "/" )[-2][ : -4 ] )

            FOLDER_OF_CAMERA_AND_LIDAR_INFO = FOLDER_OF_ROOT_KRADAR_DATASET + "{}_meta/time_info/".format( index_of_Kradar_scene_in_camera_image )

            with open( FOLDER_OF_CAMERA_AND_LIDAR_INFO + "cam-front.txt" , "r+" ) as f :
                
                list_of_camera_image_index_and_camera_image_timestamp = f.readlines()
                
                dict_of_camera_image_index_and_timestamp = {}
                
                for camera_image_index_and_timestamp in list_of_camera_image_index_and_camera_image_timestamp :
                    
                    camera_image_index = str( camera_image_index_and_timestamp.replace(" ", "").split(",")[0][ -9 : -4] )
                    
                    camera_image_timestamp = float( camera_image_index_and_timestamp.replace( " ", "").split(",")[1])
                    
                    dict_of_camera_image_index_and_timestamp[ camera_image_index ] = camera_image_timestamp 
                    
            print( "Dictionary of camera image index is : " + str( dict_of_camera_image_index_and_timestamp ))

            self.dict_of_camera_image_index_and_timestamp = dict_of_camera_image_index_and_timestamp


            with open( FOLDER_OF_CAMERA_AND_LIDAR_INFO + "os1-128.txt" , "r+" ) as f :
                
                list_of_camera_image_index_and_camera_image_timestamp = f.readlines()
                
                dict_of_lidar_index_and_timestamp = {}
                
                for camera_image_index_and_timestamp in list_of_camera_image_index_and_camera_image_timestamp :
                    
                    camera_image_index = str( camera_image_index_and_timestamp.replace(" ", "").split(",")[0][ -9 : -4] )
                    
                    camera_image_timestamp = float( camera_image_index_and_timestamp.replace( " ", "").split(",")[1])
                    
                    dict_of_lidar_index_and_timestamp[ camera_image_index ] = camera_image_timestamp 

            print( "=============================================\n\n")
            print( "Dictionary of lidar OS 128 channel timestamp is : " + str( dict_of_lidar_index_and_timestamp))

            self.dict_of_lidar_index_and_timestamp = dict_of_lidar_index_and_timestamp

        except Exception as e:

            print( "Cannot find timestamp for camera images and LiDAR point cloud selected folder because error : " + str( e ) )

            QtWidgets.QMessageBox.warning(self, 'Cannot Find Timestamp of Camera Images and LiDAR Point Cloud Timestamp', 'Please make sure Camera Images and LiDAR Point Cloud Timestamp is in Provided Folder' )


        self.logs = getImages(self.folder)
        self.numImages = len(self.logs)

        # make qitems of the image names
        self.items = [QtWidgets.QListWidgetItem(log['name']) for log in self.logs]
        for item in self.items:
            self.qlist_images.addItem(item)

        # display first image and enable Pan 
        self.cntr = 0
        self.image_viewer.enablePan(True)
        self.image_viewer.loadImage(self.logs[self.cntr]['path'])
        
        self.drivable_area_probability_mask_rgb_image = None
        self.probability_masks = None
        self.bev_drivable_area_image = None
        self.bev_drivable_area_label = None 

        self.drivable_area_image_viewer.loadImage(self.logs[self.cntr]['path'])
        self.bev_drivable_area_image_viewer.loadImage(self.logs[self.cntr]['path'])
        self.items[self.cntr].setSelected(True)
        #self.qlist_images.setItemSelected(self.items[self.cntr], True)

        # enable the next image button on the gui if multiple images are loaded
        if self.numImages > 1:
            self.next_im.setEnabled(True)

    def resizeEvent(self, evt):
        if self.cntr >= 0:
            self.image_viewer.onResize()

    def nextImg(self):
        if self.cntr < self.numImages -1:
            self.cntr += 1
            self.image_viewer.loadImage(self.logs[self.cntr]['path'])
            self.drivable_area_image_viewer.loadImage(self.logs[self.cntr]['path'])
            #self.bev_drivable_area_image_viewer.loadImage(self.logs[self.cntr]['path'])

            path_of_current_image = self.logs[self.cntr]['path']

            index_of_current_image_labelling = path_of_current_image.split("/")[-1][-9 : -4 ]

            if self.find_lidar_index_same_time_of_camera_image_front_cam( index_of_current_image_labelling ) is not None :

                self.lidar_index_for_this_camera_image = self.find_lidar_index_same_time_of_camera_image_front_cam( index_of_current_image_labelling )

                self.bev_lidar_points_visualization_in_rgb_image = self.visualize_lidar_points_visualization( self.lidar_index_for_this_camera_image )

                self.bev_drivable_area_image_viewer.loadImageFromArray( self.bev_lidar_points_visualization_in_rgb_image )

            else :

                self.bev_drivable_area_image_viewer.loadImage(self.logs[self.cntr]['path'])

            if self.bev_drivable_area_image is not None :

                self.previous_frame_drivable_area_image = self.bev_drivable_area_image

                self.previous_frame_drivable_area_image_bev_label = self.bev_drivable_area_label

                self.bev_drivable_area_previous_image_q_label_image_viewer.loadImageFromArray(self.previous_frame_drivable_area_image)

                

            self.items[self.cntr].setSelected(True)
            #self.qlist_images.setItemSelected(self.items[self.cntr], True)
            self.drivable_area_probability_mask_rgb_image = None
            self.probability_masks = None
            self.bev_drivable_area_image = None
            self.bev_drivable_area_label = None 
        else:
            QtWidgets.QMessageBox.warning(self, 'Sorry', 'No more Images!')

    def prevImg(self):
        if self.cntr > 0:
            self.cntr -= 1
            self.image_viewer.loadImage(self.logs[self.cntr]['path'])
            self.drivable_area_image_viewer.loadImage(self.logs[self.cntr]['path'])
            #self.bev_drivable_area_image_viewer.loadImage(self.logs[self.cntr]['path'])

            path_of_current_image = self.logs[self.cntr]['path']

            index_of_current_image_labelling = path_of_current_image.split("/")[-1][-9 : -4 ]

            if self.find_lidar_index_same_time_of_camera_image_front_cam( index_of_current_image_labelling ) is not None :

                self.lidar_index_for_this_camera_image = self.find_lidar_index_same_time_of_camera_image_front_cam( index_of_current_image_labelling )

                self.bev_lidar_points_visualization_in_rgb_image = self.visualize_lidar_points_visualization( self.lidar_index_for_this_camera_image )

                self.bev_drivable_area_image_viewer.loadImageFromArray( self.bev_lidar_points_visualization_in_rgb_image )

            else :

                self.bev_drivable_area_image_viewer.loadImage(self.logs[self.cntr]['path'])

            if self.bev_drivable_area_image is not None :
                self.previous_frame_drivable_area_image = self.bev_drivable_area_image

                self.previous_frame_drivable_area_image_bev_label = self.bev_drivable_area_label

                self.bev_drivable_area_previous_image_q_label_image_viewer.loadImageFromArray( self.previous_frame_drivable_area_image )
                
            self.items[self.cntr].setSelected(True)
            #self.qlist_images.setItemSelected(self.items[self.cntr], True)
            self.drivable_area_probability_mask_rgb_image = None
            self.probability_masks = None
            self.bev_drivable_area_image = None
            self.bev_drivable_area_label = None 
        else:
            QtWidgets.QMessageBox.warning(self, 'Sorry', 'No previous Image!')

    def item_click(self, item):
        self.cntr = self.items.index(item)
        self.image_viewer.loadImage(self.logs[self.cntr]['path'])
        self.drivable_area_image_viewer.loadImage( self.logs[self.cntr]['path'] )
        #self.bev_drivable_area_image_viewer.loadImage( self.logs[self.cntr][ 'path' ])

        path_of_current_image = self.logs[self.cntr]['path']

        index_of_current_image_labelling = path_of_current_image.split("/")[-1][-9 : -4 ]

        if self.find_lidar_index_same_time_of_camera_image_front_cam( index_of_current_image_labelling ) is not None :

            self.lidar_index_for_this_camera_image = self.find_lidar_index_same_time_of_camera_image_front_cam( index_of_current_image_labelling )

            self.bev_lidar_points_visualization_in_rgb_image = self.visualize_lidar_points_visualization( self.lidar_index_for_this_camera_image )

            self.bev_drivable_area_image_viewer.loadImageFromArray( self.bev_lidar_points_visualization_in_rgb_image )

        else :

            self.bev_drivable_area_image_viewer.loadImage(self.logs[self.cntr]['path'])

        if self.bev_drivable_area_image is not None :
            self.previous_frame_drivable_area_image = self.bev_drivable_area_image

            self.previous_frame_drivable_area_image_bev_label = self.bev_drivable_area_label

            self.bev_drivable_area_previous_image_q_label_image_viewer.loadImageFromArray( self.previous_frame_drivable_area_image )

        self.drivable_area_probability_mask_rgb_image = None 

    def action_line(self):
        #print( "Lets make a line on the image...")
        if self.toggle_line.isChecked():
            self.qlabel_image.setCursor(QtCore.Qt.CrossCursor)
            self.image_viewer.enablePan(False)

    def action_rect(self):
        if self.toggle_rect.isChecked():
            self.qlabel_image.setCursor(QtCore.Qt.CrossCursor)
            self.image_viewer.enablePan(False)

    def action_move(self):
        if self.toggle_move.isChecked():
            self.qlabel_image.setCursor(QtCore.Qt.OpenHandCursor)
            self.image_viewer.enablePan(True)

    def add_attention_point_button_clicked(self):

        if not self.folder:

            QtWidgets.QMessageBox.warning(self, 'You have to select Image to predict Drivable Area', 'Please select image to predict Drivable Area...')

            return
        
        else :

            if self.is_add_attention_point == True :
                self.is_add_attention_point = False 
            elif self.is_add_attention_point == False :
                self.is_add_attention_point = True
    
    def add_attention_point(self, QMouseEvent) :
        
        if self.is_add_attention_point == True :

            if QMouseEvent.button() == Qt.LeftButton :
                self.center = QMouseEvent.pos()
                x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
                actual_x_on_image = int( x*self.image_viewer.qimage.width()/self.image_viewer.qimage_scaled.width())
                actual_y_on_image = int( y*self.image_viewer.qimage.height()/self.image_viewer.qimage_scaled.height())
                #print(x,y)
                #print( "Actual point location on x and y is : " + str( actual_x_on_image ) + " " + str( actual_y_on_image ))

            # Add Point clicked on image to list of attention points for DA segmentation by DA detection Segment Anything
                self.list_of_attention_points.append( [actual_x_on_image,actual_y_on_image] )
                self.list_of_attention_point_column.addItem(str([ actual_x_on_image , actual_y_on_image ]))
                #self.list_of_attention_point_column.addItem( "{} \t ".format(len(self.list_of_attention_points) -1 ) +str([actual_x_on_image, actual_y_on_image]))
                self.list_of_attention_points_mark_on_image.append( [x,y])
            elif QMouseEvent.button() == Qt.RightButton :
                # Then attention point is for non drivable area

                #return
                self.center = QMouseEvent.pos()
                x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
                actual_x_on_image = int( x*self.image_viewer.qimage.width()/self.image_viewer.qimage_scaled.width())
                actual_y_on_image = int( y*self.image_viewer.qimage.height()/self.image_viewer.qimage_scaled.height())

                self.list_of_attention_points_non_drivable_area.append( [actual_x_on_image, actual_y_on_image ])
                self.list_of_attention_points_non_drivable_area_mark_on_image.append( [x,y] )
                self.list_of_attention_points_non_drivable_area_column.addItem( str([ actual_x_on_image , actual_y_on_image ]) )

            self.update()

        #qp = QPainter(self)
        #rect = self.rect()
        #qp.drawImage(rect, self.image_viewer.qimage_scaled, rect)
        #if self.moving:
        #    self.draw_circle(qp)
        #elif self.pressed:
        #self.draw_point(qp)
        #print( "List of Attention Point is : " + str( self.list_of_attention_points ))
        #if self.panFlag:
        #    self.pressed = QMouseEvent.pos()    # starting point of drag vector
        #    self.anchor = self.position         # save the pan position when panning starts

    #def action_add_attention_points
    def delete_attention_points(self, item):

        #print( "Deleted item is : " + str( self.list_of_attention_point_column.row( item )) )

        self.list_of_attention_points = [self.list_of_attention_points[i] for i in range( len( self.list_of_attention_points ) ) if int( i ) != int( self.list_of_attention_point_column.row( item )) ]
        self.list_of_attention_points_mark_on_image = [self.list_of_attention_points_mark_on_image[i] for i in range( len( self.list_of_attention_points_mark_on_image ) ) if int( i ) != int( self.list_of_attention_point_column.row( item )) ]
        #print( "List of attention points now are : " + str( self.list_of_attention_points ))
        self.list_of_attention_point_column.takeItem( self.list_of_attention_point_column.row( item ) )

        # Then delete mark point in attention point column
        self.update()

    def delete_attention_points_non_drivable_area(self, item):

        #print( "Deleted item is : " + str( self.list_of_attention_point_column.row( item )) )

        self.list_of_attention_points_non_drivable_area = [self.list_of_attention_points_non_drivable_area[i] for i in range( len( self.list_of_attention_points_non_drivable_area ) ) if int( i ) != int( self.list_of_attention_points_non_drivable_area_column.row( item )) ]
        self.list_of_attention_points_non_drivable_area_mark_on_image = [self.list_of_attention_points_non_drivable_area_mark_on_image[i] for i in range( len( self.list_of_attention_points_non_drivable_area_mark_on_image ) ) if int( i ) != int( self.list_of_attention_points_non_drivable_area_column.row( item )) ]
        #print( "List of attention points now are : " + str( self.list_of_attention_points ))
        self.list_of_attention_points_non_drivable_area_column.takeItem( self.list_of_attention_points_non_drivable_area_column.row( item ) )

        # Then delete mark point in attention point column
        self.update()
    
    def delete_all_attention_points(self):
        self.list_of_attention_points = []
        self.list_of_attention_points_mark_on_image = []
        self.list_of_attention_points_non_drivable_area = []
        self.list_of_attention_points_non_drivable_area_mark_on_image = []

        # Remove all list of attention points and non drivable area attention points on list_of_attention_points_column and list_of_attention_points_non_drivable_area_column

        self.list_of_attention_point_column.clear()
        self.list_of_attention_points_non_drivable_area_column.clear()

        self.update()


    def mouseReleaseEvent_for_add_attention_point(self, event):
        if ( event.button() == Qt.LeftButton ) & ( self.is_add_attention_point == True ):
            #self.revisions.append(self.image.copy())
            #qp = QPainter(self.image_viewer.qlabel_image)
            #self.draw_point(qp)
            #self.pressed = self.moclearving = False
            self.update()

    def paintEvent(self, event):

        if self.is_add_attention_point == False :
            return
        qp = QPainter(self.image_viewer.qpixmap)
        qp.drawImage(QtCore.QPoint(0, 0), self.image_viewer.qimage_scaled,
                    QtCore.QRect(self.image_viewer.position[0], self.image_viewer.position[1], self.image_viewer.qlabel_image.width(), self.image_viewer.qlabel_image.height()) )
        #print( qp )
        #rect = self.rect()
        #qp.drawImage(rect, self.image_viewer.qimage_scaled, rect)
        #if self.moving:
        #self.draw_circle(qp)
        #elif self.pressed:
        self.draw_point(qp)
        self.image_viewer.qlabel_image.setPixmap(self.image_viewer.qpixmap)
        qp.end()

    """
    def mousePressEvent(self, event):
        if self.pixmap_item is self.itemAt(event.pos()):
            sp = self.mapToScene(event.pos())
            lp = self.pixmap_item.mapFromScene(sp).toPoint()
            print(lp)
    """

    def draw_point(self, qp):

        if self.is_add_attention_point == True :

            qp.setPen(QPen(Qt.green, 5))
            #print( "Added attention point coordinate is : " + str( self.center ))
            for i, attention_point_click in enumerate( self.list_of_attention_points_mark_on_image ) :
                qp.drawEllipse(attention_point_click[0], attention_point_click[1],5,5)
                qp.drawText(attention_point_click[0], attention_point_click[1], str(i))
            
            qp.setPen(QPen(Qt.red, 5))
            for i, attention_point_click in enumerate( self.list_of_attention_points_non_drivable_area_mark_on_image ) :
                qp.drawEllipse(attention_point_click[0], attention_point_click[1],5,5)
                qp.drawText(attention_point_click[0], attention_point_click[1], "non-DA : " + str(i))
            

    def predict_drivable_area(self) :

        if len( self.list_of_attention_points ) <= 0 :

            # If there is no attention points
            # Then couldnt predict drivable area in the image

            QtWidgets.QMessageBox.warning(self, 'Cannot predict Drivable Area in Image ', 'Couldnt predict Drivable Area in image because there is no attention points selected')

            return 
        
        else :

            self.dir_of_selected_images = self.logs[self.cntr]['path']

            #print( "Directory of selected images are : " + str( self.dir_of_selected_images ))
            
            image_to_be_predicted = cv2.imread( self.dir_of_selected_images )[ : , : 1280].astype( np.uint8 )

            image_to_be_predicted = cv2.cvtColor(image_to_be_predicted, cv2.COLOR_BGR2RGB)

            #print( "Shape of image to be predicted is : " + str( image_to_be_predicted.shape ))

            #plt.imshow( image_to_be_predicted )

            #plt.show()
            
            self.drivable_area_predictor.set_image(image_to_be_predicted)

            list_of_drivable_area_label = self.list_of_attention_points + self.list_of_attention_points_non_drivable_area #[[100,600], [ 320, 540] , [640, 540] , [ 960,540 ] , [400, 600]] #[[x,540] for x in range( 0 , 1200 , 400)]

            list_of_drivable_area_label_drivable_or_not = [1 for i in self.list_of_attention_points ] + [0 for i in self.list_of_attention_points_non_drivable_area]
            #print( "List of drivable area label is : " + str( list_of_drivable_area_label ))

            masks, _, _ = self.drivable_area_predictor.predict( np.array( list_of_drivable_area_label), np.array( list_of_drivable_area_label_drivable_or_not) )

            probability_masks = masks[0].astype( np.float32 )

            self.probability_masks = probability_masks

            self.drivable_area_probability_mask_rgb_image = np.array( [[[0,255,0] if i > 0.8 else [255,255,255] for i in j ] for j in probability_masks ]).astype( np.uint8 ) 

            #print( "Mask prediction result is : " + str( probability_masks ) + " with shape of masks are : " + str( probability_masks.shape ))

            #print( "Number of drivable area is : " + str( np.sum( probability_masks )))

            # Write masking result to image

            #with open( "result_drivable__area_segmentation.png" )

            image_predicted_with_drivable_area_prediction = image_to_be_predicted.copy()

            image_predicted_with_drivable_area_prediction[ probability_masks > 0.99 ] = [0,255,0]

            for camera_image_projection_coordinate_y in range( probability_masks.shape[0] ) :

                for camera_image_projection_coordinate_x in range( probability_masks.shape[1] ) :

                    if probability_masks[ camera_image_projection_coordinate_y ][ camera_image_projection_coordinate_x ] > 0.99 :

                        image_predicted_with_drivable_area_prediction[ camera_image_projection_coordinate_y , camera_image_projection_coordinate_x ] = [ int( 10 + ( 255 - 10 )*camera_image_projection_coordinate_y/ probability_masks.shape[0]) , int( 10 + ( 255 - 10 )* camera_image_projection_coordinate_x/ probability_masks.shape[1]) , 0 ]

            #image_predicted_with_drivable_area_prediction = np.transpose( image_predicted_with_drivable_area_prediction , axes = (1,0,2) )

            #print( "Shape of Drivable Area prediction is : " + str( image_predicted_with_drivable_area_prediction.shape ) + " with Drivable Area prediction is : " + str( image_predicted_with_drivable_area_prediction ))

            # Show drivable area segmentation result on the Drivable Area Segmentation Point

            self.drivable_area_image_viewer.loadImageFromArray( image_predicted_with_drivable_area_prediction.astype( np.uint8 ) )

            #QtWidgets.QMessageBox.setInformativeText( "Finish predicting drivable area label in Images")
            
            # Give notification finished predicting Drivable Area Label in Image
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Finish predict Drivable Are in Image")
            msg.setText("Finish predicting drivable area label in Images")
            msg.exec()

            #self.save_drivable_area_label()

            #plt.imsave( "result_drivable_area_segmentation_with_many_segmentation.png" , image_predicted_with_drivable_area_prediction )

            #plt.imshow( image_predicted_with_drivable_area_prediction )

            return
        
    def find_lidar_index_same_time_of_camera_image_front_cam( self, index_of_image_camera : str):
    
        TRESHOLD_OF_TIME_DIFFERENCE_CAMERA_IMAGE_AND_LIDAR_PCD = 0.015
        
        assert index_of_image_camera in self.dict_of_camera_image_index_and_timestamp.keys()
        
        time_timestamp_of_current_image_camera = self.dict_of_camera_image_index_and_timestamp[ index_of_image_camera ]
        
        for camera_image_index, camera_image_timestamp in self.dict_of_lidar_index_and_timestamp.items() :
            
            if abs( camera_image_timestamp - time_timestamp_of_current_image_camera) <= TRESHOLD_OF_TIME_DIFFERENCE_CAMERA_IMAGE_AND_LIDAR_PCD :
                
                return str( camera_image_index )
            
        return None
    
    def visualize_lidar_points_visualization( self, index_of_lidar_points_for_this_camera ) :

        FOLDER_OF_LIDAR_PCD_KRADAR_DATASET = "/".join( self.folder.split("/")[: -2] ) + "/" + str( self.folder.split("/")[-2])[ : -3] + "lpc" + "/os1-128/"

        roi_of_drivable_area = [0, 30 ,-20,20,-2,2]

        voxel_size= [0.1 , 0.1 ]#[1,1]
  
        pcd = o3d.io.read_point_cloud(FOLDER_OF_LIDAR_PCD_KRADAR_DATASET + "os1-128_" + str( index_of_lidar_points_for_this_camera ) + ".pcd" )
        
        out_arr = np.asarray(pcd.points)  
    
        out_arr = out_arr[ ( out_arr[ : , 0 ] >= roi_of_drivable_area[0] ) & ( out_arr[ : , 0 ] <= roi_of_drivable_area[1]) & (out_arr[ : , 1 ] >= roi_of_drivable_area[2] ) & (out_arr[ : , 1] <= roi_of_drivable_area[3]) & ( out_arr[ : , 2 ] >= roi_of_drivable_area[4] ) & (out_arr[ : , 2 ] <= roi_of_drivable_area[5]) ]


        width_of_bev_map = int( (roi_of_drivable_area[3] - roi_of_drivable_area[2])/voxel_size[1])
        height_of_bev_map = int( (roi_of_drivable_area[1] - roi_of_drivable_area[0])/voxel_size[0])

        bev_drivable_area_label = np.ones((height_of_bev_map, width_of_bev_map)) * -100

        for lidar_points_around_autonomous_vehicle in out_arr :

            # Find maximum LiDAR point measurement heights in every DA grids

            try : 

                if bev_drivable_area_label[height_of_bev_map - int( lidar_points_around_autonomous_vehicle[0]/ voxel_size[1]) , int( (-1*lidar_points_around_autonomous_vehicle[1] - roi_of_drivable_area[2])/voxel_size[0]) ] < lidar_points_around_autonomous_vehicle[ 2 ] :

                    bev_drivable_area_label[height_of_bev_map - int( lidar_points_around_autonomous_vehicle[0]/ voxel_size[1]) , int( (-1*lidar_points_around_autonomous_vehicle[1] - roi_of_drivable_area[2])/voxel_size[0]) ] = lidar_points_around_autonomous_vehicle[2] # max( lidar_points_around_autonomous_vehicle[2] , )

            except :

                continue 

        # Convert LiDAR point BEV drivable area into LiDAR point BEV drivable area RGB image

        bev_drivable_area_label_rgb_images = np.array( [[[255 , 255 , 255] if i== -100 else [0 , 0 , int( 100 + (-1* i+2)/4 * ( 255 - 100 ))]  for i in j ] for j in bev_drivable_area_label] ).astype( np.uint8 )

        #print( "BEV lidar visualization in RGB images are : " + str( bev_drivable_area_label_rgb_images))


        return bev_drivable_area_label_rgb_images




    
    def visualize_bev_drivable_area_label_using_lidar( self, index_of_image_camera : str , is_return_drivable_area_in_image_and_bev = False ):
    
        index_of_lidar_point_cloud_same_time_of_camera_image = self.find_lidar_index_same_time_of_camera_image_front_cam( index_of_image_camera )

        if index_of_lidar_point_cloud_same_time_of_camera_image is None :

            # Then there is no LiDAR point cloud for the image drivable area

            QtWidgets.QMessageBox.warning(self, 'There is No LiDAR Point Cloud in Same Time with Selected Images', 'There is no LiDAR point cloud in same time with selected images')

            return None , None 
        
        FOLDER_OF_CAMERA_IMAGE_KRADAR_DATASET = self.folder

        FOLDER_OF_LIDAR_PCD_KRADAR_DATASET = "/".join( self.folder.split("/")[: -2] ) + "/" + str( self.folder.split("/")[-2])[ : -3] + "lpc" + "/os1-128/"

        roi_of_drivable_area = [0, 30 ,-20,20,-2,2]

        voxel_size= [0.5 , 0.5 ]#[1,1]
        
        name_of_camera_image_of_same_time_with_lidar_pcd = FOLDER_OF_CAMERA_IMAGE_KRADAR_DATASET + "/" + "cam-front_" + index_of_image_camera + ".png"
        
        camera_image_of_same_time_with_lidar_pcd = plt.imread( name_of_camera_image_of_same_time_with_lidar_pcd )[ : , :1280]
        
        drivable_area_label_of_image_same_time_with_lidar_pcd = self.probability_masks #plt.imread( FOLDER_OF_DRIVABLE_AREA_LABEL_KRADAR_DATASET +  "cam-front_" + index_of_camera_image_same_time_of_lidar_pcd + ".png" )#.transpose(1,0,2)

        #print( "Shape of Drivable Area Lable of Image Same Time with LiDAR PCD is : " + str( drivable_area_label_of_image_same_time_with_lidar_pcd.shape ))
        
        if drivable_area_label_of_image_same_time_with_lidar_pcd is None :
            # Then there is no drivable area label on image

            QtWidgets.QMessageBox.warning(self, 'No Drivable Area Label in Image', 'Please make Drivable Area Label in Image')

            return None , None 
        
        camera_image_with_drivable_area_label = camera_image_of_same_time_with_lidar_pcd.copy() * 255#.astype( np.uint8 )
        
        #print( "Shape of camera image of KRadar Dataset is : " + str( camera_image_of_same_time_with_lidar_pcd.shape ) + " and shape of drivable area label in camera image KRadar dataset is : " + str( drivable_area_label_of_image_same_time_with_lidar_pcd.shape ) + " with camera image is : " + str( camera_image_of_same_time_with_lidar_pcd))
        
        # Draw drivable area label on camera image same time with lidar pcd with drivable area label color
        
        camera_image_with_drivable_area_label[ drivable_area_label_of_image_same_time_with_lidar_pcd[ : , : ] > 0.8 ] = [0,255,0]
        
        #plt.figure(figsize=(20,11))
        #plt.imshow( camera_image_with_drivable_area_label.astype(np.uint8) )
        
        #plt.show()
        
        pcd = o3d.io.read_point_cloud(FOLDER_OF_LIDAR_PCD_KRADAR_DATASET + "os1-128_" + index_of_lidar_point_cloud_same_time_of_camera_image + ".pcd" )
        
        out_arr = np.asarray(pcd.points)  
    
        out_arr = out_arr[ ( out_arr[ : , 0 ] >= roi_of_drivable_area[0] ) & ( out_arr[ : , 0 ] <= roi_of_drivable_area[1]) & (out_arr[ : , 1 ] >= roi_of_drivable_area[2] ) & (out_arr[ : , 1] <= roi_of_drivable_area[3]) & ( out_arr[ : , 2 ] >= roi_of_drivable_area[4] ) & (out_arr[ : , 2 ] <= roi_of_drivable_area[5]) ]
        
        out_arr_for_projection = out_arr.transpose()
        
        lidar_points_in_camera_image = self.camera_intrinsic_parameter @ self.camera_extrinsic_transformation_matrix @ out_arr_for_projection.reshape(3,-1)
        
        mask_of_drivable_area_label = []
        
        drivable_area_lidar_point_label = drivable_area_label_of_image_same_time_with_lidar_pcd


        for lidar_point_in_image_coordinate in lidar_points_in_camera_image.transpose().astype(int) :

            lidar_point_in_image_coordinate = lidar_point_in_image_coordinate / ( lidar_point_in_image_coordinate[2] + 1e-12)

            if ( lidar_point_in_image_coordinate[1] >= 0 ) & (lidar_point_in_image_coordinate[1] < 720 ) & (lidar_point_in_image_coordinate[0] >= 0 ) & (lidar_point_in_image_coordinate[0] < 1280 ) :

                #print( "LiDAR point in image of Drivable Area is :")
                if drivable_area_lidar_point_label[ int( lidar_point_in_image_coordinate[1]), int( lidar_point_in_image_coordinate[0] )] > 0.8 :

                    mask_of_drivable_area_label.append( True )

                    #drivable_area_lidar_point_in_image_coordinate

                    #print( "Drivable area LiDAR point is : " + str( lidar_point_in_image_coordinate ))

                    continue

            mask_of_drivable_area_label.append(False)
        
        drivable_area_lidar_point = out_arr[ mask_of_drivable_area_label ].astype( int )[ : , : 2]

        lidar_point_drivable_area_in_image_coordinate = lidar_points_in_camera_image.transpose().astype(int)[ mask_of_drivable_area_label ]

        lidar_point_drivable_area_in_image_coordinate = lidar_point_drivable_area_in_image_coordinate / ( lidar_point_drivable_area_in_image_coordinate[ : , 2 ].reshape( -1  , 1 ) + 1e-12 )
            
        width_of_bev_map = int( (roi_of_drivable_area[3] - roi_of_drivable_area[2])/voxel_size[1])
        height_of_bev_map = int( (roi_of_drivable_area[1] - roi_of_drivable_area[0])/voxel_size[0])

        bev_drivable_area_label = np.zeros((height_of_bev_map, width_of_bev_map))

        bev_drivable_area_label_with_rgb_image = np.ones(( height_of_bev_map , width_of_bev_map , 3 )) * 255 

        for drivable_area_lidar_point_coordinate, drivable_area_lidar_point_projected_to_image_coordinate in zip( drivable_area_lidar_point , lidar_point_drivable_area_in_image_coordinate ) :
            
            bev_drivable_area_label[height_of_bev_map - int( drivable_area_lidar_point_coordinate[0]/ voxel_size[1]) , int( (-1*drivable_area_lidar_point_coordinate[1] - roi_of_drivable_area[2])/voxel_size[0])] = 1

            # Give drivable area label to RGB image with color based on Drivable Area image on the image location coordinate

            bev_drivable_area_label_with_rgb_image[height_of_bev_map - int( drivable_area_lidar_point_coordinate[0]/ voxel_size[1]) , int( (-1*drivable_area_lidar_point_coordinate[1] - roi_of_drivable_area[2])/voxel_size[0]) ] = [ int( 10 + (255 -10) * drivable_area_lidar_point_projected_to_image_coordinate[1] / 720 ),
                                                                                                                                                                                                                                      int( 10 + ( 255 - 10 ) * drivable_area_lidar_point_projected_to_image_coordinate[0] / 1280 ),
                                                                                                                                                                                                                                      0]

        bev_drivable_area_label[-4 : , int( width_of_bev_map/2 - 2) : int( width_of_bev_map/2 + 2)] = 1

        bev_drivable_area_label = bev_drivable_area_label.astype( bool )
        
        # Delete occluded drivable area because of objects in image camera
        
        #length_of_holes_to_covered = 2

        #structuring_holes_to_remove = np.ones((length_of_holes_to_covered,length_of_holes_to_covered))

        #structuring_holes_to_remove[0,0] = 0

        #structuring_holes_to_remove[0,length_of_holes_to_covered-1] = 0

        #structuring_holes_to_remove[length_of_holes_to_covered-1,0] = 0

        #structuring_holes_to_remove[length_of_holes_to_covered-1,length_of_holes_to_covered-1] = 0

        #bev_drivable_area_label = mahotas.close_holes(bev_drivable_area_label , structuring_holes_to_remove)
        
        #bev_drivable_area_label_with_rgb_image = np.array( [[[0,255,0] if i == True else [255,255,255] for i in j ] for j in bev_drivable_area_label] ).astype(np.uint8)
        
        width_of_bev_drivable_area_label = bev_drivable_area_label_with_rgb_image.shape[1]
        
        # Draw autonomous vehicle in BEV drivable area label
        
        #print( "Shape of BEV drivable area label with RGB image is : " + str( bev_drivable_area_label_with_rgb_image.shape))
        
        bev_drivable_area_label_with_rgb_image[int( -2/ voxel_size[1] ) : , int(width_of_bev_drivable_area_label/2) -int( 1/voxel_size[0] ) : int( width_of_bev_drivable_area_label/2 )+ int( 1 / voxel_size[0]) ] = [ 255, 0, 0]


        #bev_drivable_area_label_with_rgb_image = Image.blend( Image.fromarray(self.bev_lidar_points_visualization_in_rgb_image.astype( np.uint8 )) , Image.fromarray(bev_drivable_area_label_with_rgb_image.astype( np.uint8) ), 0.5 )

        # Combine LiDAR points visualization and drivable area visualization

        """

        for x_coordinate_of_bev_drivable_area in range( bev_drivable_area_label_with_rgb_image.shape[0]) :

            for y_coordinate_of_bev_drivable_area in range( bev_drivable_area_label_with_rgb_image.shape[1]) :

                if bev_drivable_area_label_with_rgb_image[x_coordinate_of_bev_drivable_area][y_coordinate_of_bev_drivable_area][2] == 255 :

                    bev_drivable_area_label_with_rgb_image[x_coordinate_of_bev_drivable_area][y_coordinate_of_bev_drivable_area] = self.bev_lidar_points_visualization_in_rgb_image[x_coordinate_of_bev_drivable_area][y_coordinate_of_bev_drivable_area]

        """


        return np.array( bev_drivable_area_label_with_rgb_image ), bev_drivable_area_label
    
    def generates_bev_DA_label_projected_to_lidar_pcd(self) :

        path_of_current_image = self.logs[self.cntr]['path']

        index_of_current_image_labelling = path_of_current_image.split("/")[-1][-9 : -4 ]

        image_of_DA_label_projection, bev_drivable_area_label  = self.visualize_bev_drivable_area_label_using_lidar( index_of_current_image_labelling )

        if image_of_DA_label_projection is not None :

            print( "Shape of DA Label projection is : " + str( image_of_DA_label_projection.shape ) + " with Data Types : " + str( image_of_DA_label_projection))

            self.bev_drivable_area_image_viewer.loadImageFromArray( image_of_DA_label_projection.astype( np.uint8 ) )

            self.bev_drivable_area_image = image_of_DA_label_projection

            self.bev_drivable_area_image_in_rgb_image = image_of_DA_label_projection

            if ( self.bev_drivable_area_image.shape != self.bev_lidar_points_visualization_in_rgb_image.shape ) :

                self.bev_drivable_area_image = cv2.resize(self.bev_drivable_area_image, dsize=( self.bev_lidar_points_visualization_in_rgb_image.shape[ 1] , self.bev_lidar_points_visualization_in_rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

                #self.bev_drivable_area_image = np.array( self.bev_drivable_area_image ).transpose( (1,0,2) )

                assert self.bev_drivable_area_image.shape == self.bev_lidar_points_visualization_in_rgb_image.shape , "Dimension of BEV drivable area image is : {} While dimension of BEV LiDAR points visualization is : {}".format( self.bev_drivable_area_image.shape , self.bev_lidar_points_visualization_in_rgb_image.shape )

            # Then combine BEV LiDAR Points visualization and BEV DA Label Visualizations

            for x_coordinate_of_bev_drivable_area in range( self.bev_drivable_area_image.shape[0]) :

                for y_coordinate_of_bev_drivable_area in range( self.bev_drivable_area_image.shape[1]) :

                    if self.bev_drivable_area_image[x_coordinate_of_bev_drivable_area][y_coordinate_of_bev_drivable_area][2] == 255 :

                        self.bev_drivable_area_image[x_coordinate_of_bev_drivable_area][y_coordinate_of_bev_drivable_area] = self.bev_lidar_points_visualization_in_rgb_image[x_coordinate_of_bev_drivable_area][y_coordinate_of_bev_drivable_area]

            

            self.bev_drivable_area_label = bev_drivable_area_label

            bev_drivable_area_image_with_LiDAR_visualization = self.bev_drivable_area_image.astype( np.uint8 )

            #plt.plot( bev_drivable_area_image_with_LiDAR_visualization)

            #plt.show()

            #plt.savefig( bev_drivable_area_image_with_LiDAR_visualization , "BEV_Drivable_Area_Image_with_LiDAR_Visualization.png" )

            cv2.imwrite('bev_drivable_area_image_with_LiDAR_visualization.png', bev_drivable_area_image_with_LiDAR_visualization)

            #print( "Shape of BEV Drivable Area Image with LiDAR Visualization is : " + str( bev_drivable_area_image_with_LiDAR_visualization.shape ) + " with Data Types : " + str( bev_drivable_area_image_with_LiDAR_visualization ))

            self.bev_drivable_area_image_viewer.loadImageFromArray( bev_drivable_area_image_with_LiDAR_visualization )

            # Give notification finished predicting Drivable Area Label in Image
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Finish predict BEV Drivable Area Label using LiDAR pcd")
            msg.setText("Finish predicting BEV Drivable Area Label using LiDAR pcd")
            msg.exec()

            self.save_drivable_area_label()
        
        return 

    def fix_holes_drivable_area_label(self):

        if self.probability_masks is not None:

            # Delete occluded drivable area because of objects in image camera
    
            length_of_holes_to_covered = 2 #2

            structuring_holes_to_remove = np.ones((length_of_holes_to_covered,length_of_holes_to_covered))

            probability_masks = mahotas.close_holes(self.probability_masks.astype( bool ) , structuring_holes_to_remove)

            self.probability_masks = probability_masks.astype(np.float32)

            self.drivable_area_probability_mask_rgb_image = np.array( [[[0,255,0] if i > 0.8 else [255,255,255] for i in j ] for j in probability_masks ]).astype( np.uint8 ) 

            # Show fixed Drivable Area label in Image

            image_to_be_predicted = cv2.imread( self.dir_of_selected_images )[ : , : 1280].astype( np.uint8 )

            image_to_be_predicted = cv2.cvtColor(image_to_be_predicted, cv2.COLOR_BGR2RGB)

            image_predicted_with_drivable_area_prediction = image_to_be_predicted.copy()

            image_predicted_with_drivable_area_prediction[ probability_masks > 0.99 ] = [0,255,0]

            self.drivable_area_image_viewer.loadImageFromArray( image_predicted_with_drivable_area_prediction.astype( np.uint8 ) )

            # Give notification finished predicting Drivable Area Label in Image
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Finish fixed Drivable Area label in camera")
            msg.setText("Finish fixing drivable area label in Images")
            msg.exec()

    def delete_drivable_area_label_MousePressedButton(self, QMouseEvent ) :

        if self.drivable_area_image_viewer.qimage_scaled is not None :

            # Then there is Drivable Area predicted on image

            if QMouseEvent.button() == Qt.RightButton :

                if self.delete_drivable_area_on_image_start_point is None :

                    self.center = QMouseEvent.pos()
                    x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
                    actual_x_on_image = int( x*self.drivable_area_image_viewer.qimage.width()/self.drivable_area_image_viewer.qimage_scaled.width())
                    actual_y_on_image = int( y*self.drivable_area_image_viewer.qimage.height()/self.drivable_area_image_viewer.qimage_scaled.height())

                    self.delete_drivable_area_on_image_start_point = [ actual_x_on_image , actual_y_on_image ]

    def delete_drivable_area_label_MouseReleaseButton( self, QMouseEvent ) :

        if QMouseEvent.button() == Qt.RightButton :

            if self.delete_drivable_area_on_image_start_point is not None :

                # Then delete drivable area predicted by Deep Learning on Drivable Area Image 

                self.center = QMouseEvent.pos()
                x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
                actual_x_on_image = int( x*self.drivable_area_image_viewer.qimage.width()/self.drivable_area_image_viewer.qimage_scaled.width())
                actual_y_on_image = int( y*self.drivable_area_image_viewer.qimage.height()/self.drivable_area_image_viewer.qimage_scaled.height())

                new_image_drivable_area_probability_mask_on_image = self.drivable_area_probability_mask_rgb_image.copy()

                new_image_drivable_area_probability_mask_on_image = cv2.rectangle(new_image_drivable_area_probability_mask_on_image , self.delete_drivable_area_on_image_start_point, [actual_x_on_image , actual_y_on_image], (255 , 255 , 255), thickness = -1 )

                image_to_be_predicted = cv2.imread( self.dir_of_selected_images )[ : , : 1280].astype( np.uint8 )

                image_to_be_predicted = cv2.cvtColor(image_to_be_predicted, cv2.COLOR_BGR2RGB)

                for camera_image_projection_coordinate_y in range( new_image_drivable_area_probability_mask_on_image.shape[0] ) :

                    for camera_image_projection_coordinate_x in range( new_image_drivable_area_probability_mask_on_image.shape[1] ) :

                        if new_image_drivable_area_probability_mask_on_image[ camera_image_projection_coordinate_y ][ camera_image_projection_coordinate_x ][0] == 0 :

                            image_to_be_predicted[ camera_image_projection_coordinate_y , camera_image_projection_coordinate_x ] = [ int( 10 + ( 255 - 10 )*camera_image_projection_coordinate_y/ new_image_drivable_area_probability_mask_on_image.shape[0]) , int( 10 + ( 255 - 10 )* camera_image_projection_coordinate_x/ new_image_drivable_area_probability_mask_on_image.shape[1]) , 0 ]

                self.drivable_area_image_viewer.loadImageFromArray( image_to_be_predicted.astype( np.uint8 ) )

                self.drivable_area_probability_mask_rgb_image = new_image_drivable_area_probability_mask_on_image.astype( np.uint8 )

                self.probability_masks = np.array( [[ 0 if ( i == [255 , 255 , 255]).all() else 1 for i in j ] for j in self.drivable_area_probability_mask_rgb_image.copy() ] )

                # Then delete start point of Deleting Drivable Area Label on Image Start Point

                self.delete_drivable_area_on_image_start_point = None 

                # Give notification finished predicting Drivable Area Label in Image
                msg = QtWidgets.QMessageBox()
                msg.setWindowTitle("Finish delete Drivable Area label in image")
                msg.setText( "Finish deleteing non- Drivable Area Label in Image")
                msg.exec()
                    

                







    def add_bev_drivable_area_label_button_clicked(self):

        if self.is_add_drivable_area_in_bev_drivable_area == True :

            self.is_add_drivable_area_in_bev_drivable_area = False 
        
        elif self.is_add_drivable_area_in_bev_drivable_area == False :

            self.is_add_drivable_area_in_bev_drivable_area = True

            # Then also make Labelling Tools doesnt have delete drivable area in bev drivable area label mode
            self.is_delete_drivable_area_in_bev_drivable_area = False 

    def add_drivable_area_in_bev_drivable_area_MousePressedButton(self, QMouseEvent ) :

        if self.is_add_drivable_area_in_bev_drivable_area == True :

            if QMouseEvent.button() == Qt.LeftButton :
                 
                 if self.add_bev_drivable_area_start_point is None :
                    self.center = QMouseEvent.pos()
                    x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
                    actual_x_on_image = int( x*self.bev_drivable_area_image_viewer.qimage.width()/self.bev_drivable_area_image_viewer.qimage_scaled.width())
                    actual_y_on_image = int( y*self.bev_drivable_area_image_viewer.qimage.height()/self.bev_drivable_area_image_viewer.qimage_scaled.height())
                    
                    actual_x_on_bev_label_rgb_image = int( x*self.bev_drivable_area_image_in_rgb_image.shape[1]/self.bev_drivable_area_image_viewer.qimage_scaled.width())
                    actual_y_on_bev_label_rgb_image = int( y*self.bev_drivable_area_image_in_rgb_image.shape[0]/ self.bev_drivable_area_image_viewer.qimage_scaled.height())

                    self.add_bev_drivable_area_start_point = [ actual_x_on_image , actual_y_on_image ]

                    self.add_bev_drivable_area_start_point_in_bev_label = [ actual_x_on_bev_label_rgb_image , actual_y_on_bev_label_rgb_image]

        elif self.is_delete_drivable_area_in_bev_drivable_area == True :

            if QMouseEvent.button() == Qt.LeftButton :
                 
                 if self.delete_bev_drivable_area_start_point is None :
                    self.center = QMouseEvent.pos()
                    x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
                    actual_x_on_image = int( x*self.bev_drivable_area_image_viewer.qimage.width()/self.bev_drivable_area_image_viewer.qimage_scaled.width())
                    actual_y_on_image = int( y*self.bev_drivable_area_image_viewer.qimage.height()/self.bev_drivable_area_image_viewer.qimage_scaled.height())

                    actual_x_on_bev_label_rgb_image = int( x*self.bev_drivable_area_image_in_rgb_image.shape[1]/self.bev_drivable_area_image_viewer.qimage_scaled.width())
                    actual_y_on_bev_label_rgb_image = int( y*self.bev_drivable_area_image_in_rgb_image.shape[0]/ self.bev_drivable_area_image_viewer.qimage_scaled.height())

                    self.delete_bev_drivable_area_start_point = [ actual_x_on_image , actual_y_on_image ]

                    self.delete_bev_drivable_area_start_point_in_bev_label = [ actual_x_on_bev_label_rgb_image , actual_x_on_bev_label_rgb_image ]

    def add_drivable_area_in_bev_drivable_area_MouseReleasedButton(self, QMouseEvent ) :

        if self.is_add_drivable_area_in_bev_drivable_area == True :

            if self.add_bev_drivable_area_start_point is not None :

                # Then add additional drivable area in BEV drivable area

                self.center = QMouseEvent.pos()
                x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
                actual_x_on_image = int( x*self.bev_drivable_area_image_viewer.qimage.width()/self.bev_drivable_area_image_viewer.qimage_scaled.width())
                actual_y_on_image = int( y*self.bev_drivable_area_image_viewer.qimage.height()/self.bev_drivable_area_image_viewer.qimage_scaled.height())

                actual_x_on_bev_label_rgb_image = int( x*self.bev_drivable_area_image_in_rgb_image.shape[1]/self.bev_drivable_area_image_viewer.qimage_scaled.width())
                actual_y_on_bev_label_rgb_image = int( y*self.bev_drivable_area_image_in_rgb_image.shape[0]/ self.bev_drivable_area_image_viewer.qimage_scaled.height())

                # Then draw additional drivable area point

                # Blue color in BGR 
                color = (0, 255, 0) 

                # Line thickness of 2 px 
                thickness = -1

                # Using cv2.rectangle() method 
                # Draw a rectangle with blue line borders of thickness of 2 px 

                new_image_drivable_area_probability_mask_rgb_image = self.bev_drivable_area_image.copy()

                #print( "Adding bev drivable area label in image from coordinate : {} to coordinate : {}".format( self.add_bev_drivable_area_start_point , [ actual_x_on_image , actual_y_on_image ]))

                new_image_drivable_area_probability_mask_rgb_image = cv2.rectangle(new_image_drivable_area_probability_mask_rgb_image , self.add_bev_drivable_area_start_point, [actual_x_on_image , actual_y_on_image], color, thickness) 

                new_image_drivable_area_probability_mask_rgb_image_DA_label = cv2.rectangle(self.bev_drivable_area_image_in_rgb_image , self.add_bev_drivable_area_start_point_in_bev_label, [actual_x_on_bev_label_rgb_image, actual_y_on_bev_label_rgb_image], color, thickness) 

                self.bev_drivable_area_image = new_image_drivable_area_probability_mask_rgb_image

                self.bev_drivable_area_image_in_rgb_image = new_image_drivable_area_probability_mask_rgb_image_DA_label

                #self.bev_drivable_area_label = np.array( [ [False if (( (i[0] == 0 ) & ( i[1] ==0 )) | ( i == [255,255,255]).all() ) else True for i in j ] for j in self.bev_drivable_area_image_in_rgb_image ] )

                self.bev_drivable_area_label = np.array( [ [False if (( (i[0] == 0 ) & ( i[1] ==0 )) | ( (i[0] > 0) & (i[1] < 255 ) ) | ( i == [255,255,255]).all() ) else True for i in j ] for j in self.bev_drivable_area_image_in_rgb_image ] )

                self.bev_drivable_area_image_viewer.loadImageFromArray( self.bev_drivable_area_image )
                
                self.add_bev_drivable_area_start_point = None 

                # Give notification finished predicting Drivable Area Label in Image
                #msg = QtWidgets.QMessageBox()
                #msg.setWindowTitle("Finish Adding BEV Drivable Area Label")
                #msg.setText("Succes adding BEV drivable area label to image" )
                #msg.exec()

        elif self.is_delete_drivable_area_in_bev_drivable_area == True :

            if self.delete_bev_drivable_area_start_point is not None :

                # Then add additional drivable area in BEV drivable area

                self.center = QMouseEvent.pos()
                x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
                actual_x_on_image = int( x*self.bev_drivable_area_image_viewer.qimage.width()/self.bev_drivable_area_image_viewer.qimage_scaled.width())
                actual_y_on_image = int( y*self.bev_drivable_area_image_viewer.qimage.height()/self.bev_drivable_area_image_viewer.qimage_scaled.height())

                
                actual_x_on_bev_label_rgb_image = int( x*self.bev_drivable_area_image_in_rgb_image.shape[1]/self.bev_drivable_area_image_viewer.qimage_scaled.width())
                actual_y_on_bev_label_rgb_image = int( y*self.bev_drivable_area_image_in_rgb_image.shape[0]/ self.bev_drivable_area_image_viewer.qimage_scaled.height())

                # Then draw additional drivable area point

                # Blue color in BGR 
                color = (255, 255, 255) 

                # Line thickness of 2 px 
                thickness = -1

                # Using cv2.rectangle() method 
                # Draw a rectangle with blue line borders of thickness of 2 px 

                new_image_drivable_area_probability_mask_rgb_image = self.bev_drivable_area_image.copy()

                #print( "Adding bev drivable area label in image from coordinate : {} to coordinate : {}".format( self.add_bev_drivable_area_start_point , [ actual_x_on_image , actual_y_on_image ]))

                new_image_drivable_area_probability_mask_rgb_image = cv2.rectangle(new_image_drivable_area_probability_mask_rgb_image , self.delete_bev_drivable_area_start_point, [actual_x_on_image , actual_y_on_image], color, thickness)

                new_image_drivable_area_probability_mask_rgb_image_DA_label = cv2.rectangle(self.bev_drivable_area_image_in_rgb_image , self.delete_bev_drivable_area_start_point_in_bev_label, [actual_x_on_bev_label_rgb_image, actual_y_on_bev_label_rgb_image], color, thickness)  

                self.bev_drivable_area_image = new_image_drivable_area_probability_mask_rgb_image

                self.bev_drivable_area_image_in_rgb_image = new_image_drivable_area_probability_mask_rgb_image_DA_label

                self.bev_drivable_area_label = np.array( [ [False if ( ((i[0] == 0) & ( i[1] == 0)) | (( i[0] > 0 ) & ( i[1] < 255 )) |( i == [ 255, 255 , 255]).all()) else True for i in j ] for j in self.bev_drivable_area_image_in_rgb_image ] )

                self.bev_drivable_area_image_viewer.loadImageFromArray( self.bev_drivable_area_image )
                
                self.delete_bev_drivable_area_start_point = None 

                # Give notification finished predicting Drivable Area Label in Image
                #msg = QtWidgets.QMessageBox()
                #msg.setWindowTitle("Finish Deleting BEV Drivable Area Label")
                #msg.setText("Succes deleting specified BEV drivable area label to image" )
                #msg.exec()


    def delete_bev_drivable_area_label_button_clicked(self):

        if self.is_delete_drivable_area_in_bev_drivable_area == True :

            self.is_add_drivable_area_in_bev_drivable_area = False 
        
        elif self.is_delete_drivable_area_in_bev_drivable_area == False :

            self.is_delete_drivable_area_in_bev_drivable_area = True

            self.is_add_drivable_area_in_bev_drivable_area = False 
                     

    def copy_bev_drivable_area_label_from_previous_lidar_frame_label( self ) :

        if self.previous_frame_drivable_area_image_bev_label is None :

            QtWidgets.QMessageBox.warning(self, 'Cannot Load Drivable Area Label from Previous LiDAR Frame', 'Please make sure Labelling BEV Drivable Area in previous LiDAR frame first' )

        else :

            self.bev_drivable_area_label = self.previous_frame_drivable_area_image_bev_label

            bev_drivable_area_label_in_rgb_image = np.array( [[[ 0 , 255 , 0] if i == True else [ 255 , 255 , 255] for i in j  ] for j in self.bev_drivable_area_label ] ).astype( np.uint8)

            # Visualize previous BEV DA Label to current LiDAR frame visualization

            bev_drivable_area_label_in_lidar_point_visualization_size = cv2.resize( bev_drivable_area_label_in_rgb_image , dsize= ( self.bev_drivable_area_image.shape[1] , self.bev_drivable_area_image.shape[0]) , interpolation=cv2.INTER_NEAREST)

            assert (bev_drivable_area_label_in_lidar_point_visualization_size.shape == self.bev_drivable_area_image.shape)#.all()

            for bev_drivable_area_label_x_coordinate in range( self.bev_drivable_area_image.shape[0] ) :

                for bev_drivable_area_label_y_coordinate in range( self.bev_drivable_area_image.shape[1] ) :

                    if bev_drivable_area_label_in_lidar_point_visualization_size[ bev_drivable_area_label_x_coordinate ][ bev_drivable_area_label_y_coordinate ][0] == 255 :

                        # Then the grid in BEV_Drivabel_Area_Label_X_Coordinate and BEV_Drivable_Area_Label_Y_Coordinate is not DA area

                        bev_drivable_area_label_in_lidar_point_visualization_size[ bev_drivable_area_label_x_coordinate ][ bev_drivable_area_label_y_coordinate ] = self.bev_drivable_area_image[ bev_drivable_area_label_x_coordinate ][ bev_drivable_area_label_y_coordinate ]


            self.bev_drivable_area_image = bev_drivable_area_label_in_lidar_point_visualization_size.astype( np.uint8 )
            
            self.bev_drivable_area_image_in_rgb_image = bev_drivable_area_label_in_lidar_point_visualization_size.astype( np.uint8 )

            self.bev_drivable_area_image_viewer.loadImageFromArray( self.bev_drivable_area_image )

            # Give notification finished predicting Drivable Area Label in Image
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Finish Copying BEV Drivable Area Label from Previous LiDAR Frame")
            msg.setText("Succes copied BEV Drivable Area Label from Previous LiDAR Scene" )
            msg.exec()


    def load_bev_drivable_area_label_from_folder( self ) :

        dir_folder_drivable_area_image = "/".join( self.folder.split("/")[ : -1]) + "/BEV_DA_Label_Result_" + str( self.folder.split("/")[-1] )

        #os.makedirs( dir_folder_drivable_area_image , exist_ok= True )

        #dir_drivable_area_image = dir_folder_drivable_area_image + "/" + self.logs[self.cntr]["path"].split("/")[-1]

        name_of_BEV_DA_Label_Frame_dir = dir_folder_drivable_area_image + "/" + self.logs[self.cntr]["path"].split("/")[-1]

        if os.path.exists( name_of_BEV_DA_Label_Frame_dir ) == False :

            QtWidgets.QMessageBox.warning(self, 'Cannot Find BEV Drivable Area Label Image from Folder', 'Please make sure there is BEV Drivable Area Label in Folder ' + str( name_of_BEV_DA_Label_Frame_dir ))


        else :
            # The load BEV DA Label from BEV DA Labelling result folder

            bev_DA_label_from_folder_image = 255 * plt.imread( name_of_BEV_DA_Label_Frame_dir )#.astype( bool )

            bev_DA_label_from_folder_image = np.array( [[True if i[0] > 250 else False for i in j] for j in bev_DA_label_from_folder_image ] )

            self.bev_drivable_area_label = bev_DA_label_from_folder_image #self.previous_frame_drivable_area_image_bev_label

            #print( "BEV DA Label is : " + str( self.bev_drivable_area_label ) + " with BEV DA Label shape : " + str( self.bev_drivable_area_label.shape ) + " with maximum red value in RGB value : " + str( self.bev_drivable_area_label[ : , : , 0].max()))

            bev_drivable_area_label_in_rgb_image = np.array( [[[ 0 , 255 , 0] if i == True else [ 255 , 255 , 255] for i in j] for j in self.bev_drivable_area_label ] ).astype( np.uint8 )

            # Visualize previous BEV DA Label to current LiDAR frame visualization

            bev_drivable_area_label_in_lidar_point_visualization_size = cv2.resize( bev_drivable_area_label_in_rgb_image , dsize= ( self.bev_drivable_area_image.shape[1] , self.bev_drivable_area_image.shape[0]) , interpolation=cv2.INTER_NEAREST)

            assert (bev_drivable_area_label_in_lidar_point_visualization_size.shape == self.bev_drivable_area_image.shape)

            for bev_drivable_area_label_x_coordinate in range( self.bev_drivable_area_image.shape[0] ) :

                for bev_drivable_area_label_y_coordinate in range( self.bev_drivable_area_image.shape[1] ) :

                    if bev_drivable_area_label_in_lidar_point_visualization_size[ bev_drivable_area_label_x_coordinate ][ bev_drivable_area_label_y_coordinate ][0] == 255 :

                        # Then the grid in BEV_Drivabel_Area_Label_X_Coordinate and BEV_Drivable_Area_Label_Y_Coordinate is not DA area

                        bev_drivable_area_label_in_lidar_point_visualization_size[ bev_drivable_area_label_x_coordinate ][ bev_drivable_area_label_y_coordinate ] = self.bev_drivable_area_image[ bev_drivable_area_label_x_coordinate ][ bev_drivable_area_label_y_coordinate ]


            self.bev_drivable_area_image = bev_drivable_area_label_in_lidar_point_visualization_size.astype( np.uint8 )

            self.bev_drivable_area_image_in_rgb_image = bev_drivable_area_label_in_lidar_point_visualization_size.astype( np.uint8 )

            self.bev_drivable_area_image_viewer.loadImageFromArray( self.bev_drivable_area_image )

            # Give notification finished predicting Drivable Area Label in Image
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Finish Loading BEV Drivable Area Label from Folder")
            msg.setText("Succes loaded BEV Drivable Area Label from Folder : " + str( name_of_BEV_DA_Label_Frame_dir ) )
            msg.exec()





        
    def save_drivable_area_label(self) :

        #self.drivable_area_probability_mask_rgb_image = self.bev_drivable_area_image_in_rgb_image

        if self.drivable_area_probability_mask_rgb_image is not None :

            dir_folder_drivable_area_image = "/".join( self.folder.split("/")[ : -1]) + "/DA_Label_Result_" + str( self.folder.split("/")[-1] )

            os.makedirs( dir_folder_drivable_area_image , exist_ok= True )

            dir_drivable_area_image = dir_folder_drivable_area_image + "/" + self.logs[self.cntr]["path"].split("/")[-1]

            #drivable_area_rgb_image = np.array( [[[0,255,0] for i in j if i > 0.8  else [255,255,255]] for j in self.drivable_area_probability_masks ]).astype(np.uint8)

            plt.imsave( dir_drivable_area_image , self.drivable_area_probability_mask_rgb_image )

            #QtWidgets.QMessageBox.setInformativeText("Succes Save drivable area image to : " + str( dir_drivable_area_image ))

            # Give notification finished predicting Drivable Area Label in Image
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Finish save Drivable Area Label in Image")
            msg.setText("Succes Save drivable area image to : " + str( dir_drivable_area_image ))
            msg.exec()

        if self.bev_drivable_area_label is not None :

            dir_folder_drivable_area_image = "/".join( self.folder.split("/")[ : -1]) + "/BEV_DA_Label_Result_" + str( self.folder.split("/")[-1] )

            os.makedirs( dir_folder_drivable_area_image , exist_ok= True )

            dir_drivable_area_image = dir_folder_drivable_area_image + "/" + self.logs[self.cntr]["path"].split("/")[-1]

            #drivable_area_rgb_image = np.array( [[[0,255,0] for i in j if i > 0.8  else [255,255,255]] for j in self.drivable_area_probability_masks ]).astype(np.uint8)

            plt.imsave( dir_drivable_area_image , self.bev_drivable_area_label )

            #QtWidgets.QMessageBox.setInformativeText("Succes Save drivable area image to : " + str( dir_drivable_area_image ))

            # Give notification finished predicting Drivable Area Label in Image
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Finish save Bird Eye View Drivable Area Label in Image")
            msg.setText("Succes Save drivable area image to : " + str( dir_drivable_area_image ))
            msg.exec()
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Cleanlooks"))
    app.setPalette(QtWidgets.QApplication.style().standardPalette())
    parentWindow = Iwindow(None)
    sys.exit(app.exec_())

if __name__ == "__main__":
    #print __doc__
    main()
