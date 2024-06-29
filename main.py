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

import sys, os

from PyQt5.QtCore import QLibraryInfo

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

DIR_OF_SEGMENT_ANYTHING_MODEL = "/home/ofel04/Downloads/sam_vit_l_0b3195.pth"

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
        
        self.is_add_attention_point : bool = False
        self.list_of_attention_points = []
        self.list_of_attention_points_mark_on_image = []
        self.is_delete_attention_point : bool = False  

        self.list_of_attention_points_non_drivable_area = []
        self.list_of_attention_points_non_drivable_area_mark_on_image = []

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
        self.image_viewer.qlabel_image.mousePressEvent= self.add_attention_point
        self.image_viewer.qlabel_image.mouseReleaseEvent= self.mouseReleaseEvent_for_add_attention_point
        #self.list_of_attention_point_column.mousePressEvent = self.list_of_attention_point_column_check_if_left_mouse_pressed
        self.list_of_attention_point_column.itemClicked.connect(self.delete_attention_points)
        self.list_of_attention_points_non_drivable_area_column.itemClicked.connect(self.delete_attention_points_non_drivable_area)
        #self.qlist
        self.generate_DA_label.clicked.connect(self.predict_drivable_area )
        self.save_Drivable_Area_label.clicked.connect( self.save_drivable_area_label )
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
        
        self.drivable_area_image_viewer.loadImage(self.logs[self.cntr]['path'])
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
            self.items[self.cntr].setSelected(True)
            #self.qlist_images.setItemSelected(self.items[self.cntr], True)
            self.drivable_area_probability_mask_rgb_image = None
        else:
            QtWidgets.QMessageBox.warning(self, 'Sorry', 'No more Images!')

    def prevImg(self):
        if self.cntr > 0:
            self.cntr -= 1
            self.image_viewer.loadImage(self.logs[self.cntr]['path'])
            self.drivable_area_image_viewer.loadImage(self.logs[self.cntr]['path'])
            self.items[self.cntr].setSelected(True)
            #self.qlist_images.setItemSelected(self.items[self.cntr], True)
            self.drivable_area_probability_mask_rgb_image = None
        else:
            QtWidgets.QMessageBox.warning(self, 'Sorry', 'No previous Image!')

    def item_click(self, item):
        self.cntr = self.items.index(item)
        self.image_viewer.loadImage(self.logs[self.cntr]['path'])
        self.drivable_area_image_viewer.loadImage( self.logs[self.cntr]['path'] )
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
        self.list_of_attention_point_non_drivable_area_column.clear()

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

            self.drivable_area_probability_mask_rgb_image = np.array( [[[0,255,0] if i > 0.8 else [255,255,255] for i in j ] for j in probability_masks ]).astype( np.uint8 ) 

            #print( "Mask prediction result is : " + str( probability_masks ) + " with shape of masks are : " + str( probability_masks.shape ))

            #print( "Number of drivable area is : " + str( np.sum( probability_masks )))

            # Write masking result to image

            #with open( "result_drivable__area_segmentation.png" )

            image_predicted_with_drivable_area_prediction = image_to_be_predicted.copy()

            image_predicted_with_drivable_area_prediction[ probability_masks > 0.99 ] = [0,255,0]

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

            self.save_drivable_area_label()

            #plt.imsave( "result_drivable_area_segmentation_with_many_segmentation.png" , image_predicted_with_drivable_area_prediction )

            #plt.imshow( image_predicted_with_drivable_area_prediction )

            return
        
    def save_drivable_area_label(self) :

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
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Cleanlooks"))
    app.setPalette(QtWidgets.QApplication.style().standardPalette())
    parentWindow = Iwindow(None)
    sys.exit(app.exec_())

if __name__ == "__main__":
    #print __doc__
    main()