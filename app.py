import argparse
import cv2
import numpy as np
from helper import handle_pose, preprocessing
from inference import Network
from t_shirt_coords import TShirt, COMPENSATE_HEIGHT_TSHIRT


TShirt = TShirt()

# Global dictionary for Human coordinates
HUMAN_COORDS = {
    'NECK': 0,
    'LEFT_SHOULDER': 0,
    'RIGHT_SHOULDER': 0,
    'LEFT_HIP': 0,
    'RIGHT_HIP': 0
}


def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    s_desc = "The location of the shirt image"
    m_desc = "The location of the model XML file"

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument("-i", help=i_desc, required=True)
    required.add_argument("-s", help=s_desc, required=True)
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    args = parser.parse_args()

    return args


def get_mask(processed_output):
    '''
    For debugging
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    '''
    # Create an empty array for other color channels of mask
    empty = np.zeros(processed_output.shape)
    # Stack to make a Green mask where text detected
    mask = np.dstack((empty, processed_output, empty))

    return mask


def get_coordinates_for_human(heatmaps):
    # Get only pose detections above 0.3 confidence
    coordinates_list = np.argwhere(heatmaps>0.3)

    max_x = max([coordinates[1] for coordinates in coordinates_list]) 
    max_y = max([coordinates[0] for coordinates in coordinates_list])
    min_x = min([coordinates[1] for coordinates in coordinates_list])
    min_y = min([coordinates[0] for coordinates in coordinates_list])

    # Center point
    cX = (max_x - min_x)//2 + min_x
    cY = (max_y-min_y)//2 + min_y
    return cX,cY

def create_output_image(image, output, args):
    '''
    Using the model type, input image, and processed output,
    creates an output image showing the result of inference.
    '''
    global HUMAN_COORDS
    # Remove final part of output not used for heatmaps
    output = output[:-1]

    HUMAN_COORDS['LEFT_SHOULDER'] = get_coordinates_for_human(output[2])
    HUMAN_COORDS['RIGHT_SHOULDER'] = get_coordinates_for_human(output[5])
    HUMAN_COORDS['LEFT_HIP'] = get_coordinates_for_human(output[8])
    HUMAN_COORDS['NECK'] = get_coordinates_for_human(output[1])
    HUMAN_COORDS['RIGHT_HIP'] = get_coordinates_for_human(output[11])
    # left_arm = get_coordinates_for_human(output[3])
    # right_arm = get_coordinates_for_human(output[6])

    """
    For debugging
    cv2.circle(image, HUMAN_COORDS['LEFT_SHOULDER'], 7, (255, 255, 255), -1)
    cv2.circle(image, HUMAN_COORDS['RIGHT_SHOULDER'], 7, (255, 255, 255), -1)
    cv2.circle(image, HUMAN_COORDS['LEFT_HIP'], 7, (255, 255, 255), -1)
    cv2.circle(image, HUMAN_COORDS['NECK'], 7, (255, 255, 255), -1)
    cv2.circle(image, HUMAN_COORDS['RIGHT_HIP'], 7, (255, 255, 255), -1)
    """
    TShirt.resize_tshirt(HUMAN_COORDS)
    resized_tshirt = TShirt.get_resized_tshirt()
    TShirt.update_tshirt_coords(resized_tshirt)

    tshirt_width = resized_tshirt.shape[1] - 1
    tshirt_heigth = resized_tshirt.shape[0] - 1

    HUMAN_COORDS['NECK'] = (HUMAN_COORDS['NECK'][0],
                            int(HUMAN_COORDS['NECK'][1] - HUMAN_COORDS['NECK'][1]*COMPENSATE_HEIGHT_TSHIRT))

    # Values to get the same coordinates for the human tshirt
    dif_x = HUMAN_COORDS['NECK'][0] - TShirt.TSHIRT_COORDS['NECK'][0]
    dif_y = HUMAN_COORDS['NECK'][1] - TShirt.TSHIRT_COORDS['NECK'][1]

    for y in range(tshirt_heigth):
        for x in range(tshirt_width):
            if cv2.pointPolygonTest(TShirt.contours[0],(x,y),True) > 0:
                image[y+dif_y, x+dif_x] = resized_tshirt[y,x]

    """
    For debugging
    # Sum along the "class" axis
    output = np.sum(output, axis=0)
    # Get semantic mask
    pose_mask = get_mask(output)
    # Combine with original image
    image = image + pose_mask """
    return image

def perform_inference(args):
    '''
    Performs inference on an input image, given a model.
    '''
    # Create a Network for using the Inference Engine
    inference_network = Network()
    # Load the model in the network, and obtain its input shape
    n, c, h, w = inference_network.load_model(args.m, args.d, args.c)

    # Read the input image
    image = cv2.imread(args.i)

    ### TODO: Preprocess the input image
    preprocessed_image = preprocessing(image, h, w)

    # Perform synchronous inference on the image
    inference_network.sync_inference(preprocessed_image)

    # Obtain the output of the inference request
    output = inference_network.extract_output()

    ### TODO: Handle the output of the network, based on args.t
    ### Note: This will require using `handle_output` to get the correct
    ###       function, and then feeding the output to that function.
    processed_output = handle_pose(output, image.shape)

    # Create an output image based on network
    output_image = create_output_image(image, processed_output,args)

    # Save down the resulting image
    print("You look good!")
    cv2.imwrite("outputs/tshirt_try_on.png", output_image)


def main():
    args = get_args()
    TShirt.calculate_coords(args.s)
    perform_inference(args)


if __name__ == "__main__":
    main()
