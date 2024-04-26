import xml.etree.ElementTree as ET
from src.utils import *
from src.dataset_balancing import *
import shutil

def get_lane_lines_from_xml(path: str, label_names: list) -> list:
    tree = ET.parse(path)
    root = tree.getroot()

    frames_count = int(root.find('.//meta/job/size').text)
    width = int(root.find('.//original_size/width').text)
    height = int(root.find('.//original_size/height').text)
    shape = np.array([width, height])

    line_batches = [[] for _ in range(frames_count)]

    track_elements = root.findall('.//track')

    label_names_dict_inversed = get_label_names_dict_inversed_from_label_names(label_names)

    for track in track_elements:
        track_id = track.attrib['id']
        # label = int(label_names_dict_inversed[str(track.attrib['label'])])
        label = str(track.attrib['label'])

        polylines = track.findall('.//polyline')

        for polyline in polylines:
            

            frame = int(polyline.attrib['frame'])
            points_str = polyline.attrib['points']
            
            points = re.findall(r"[-+]?(?:\d*\.*\d+)", points_str)
            points = np.array([float(num_str) for num_str in points])

            points = points.reshape([-1, 2])
            points_n = points / shape

            lane_line = LaneLine(points, label, points_n)

            line_batches[frame].append(lane_line)
    
    return line_batches


def get_frames_count_from_xml(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frames_count = int(root.find('.//meta/job/size').text)
    return frames_count


def from_masks_to_yolo(masks: list, output_file_path: str):
    output_string = ""
    for mask in masks:
        label = mask.label
        points_n = mask.points_n

        output_string += label + " "
        for pid, point in enumerate(points_n):
            output_string += str(point[0]) + " " + str(point[1])
            output_string += " " if pid < (len(points_n) - 1) else "\n"
        
    with open(output_file_path, 'w') as out:
            out.write(output_string)


def from_mask_batches_to_yolo(mask_batches: list, output_path: str, random_names=None):
    for idx, masks in enumerate(mask_batches):
        file_path = os.path.join(output_path, f"{random_names[idx]}.txt")
        from_masks_to_yolo(masks, file_path)


def from_xml_to_yolo(xml_file_path: str, output_path: str, label_names: str, images_shape=(1920, 1080), random_names=None):
    lane_line_batches = get_lane_lines_from_xml(xml_file_path, label_names)
    lane_mask_batches = LaneMask.from_line_batches_to_mask_batches(lane_line_batches, images_shape)

    from_mask_batches_to_yolo(lane_mask_batches, output_path, random_names)


def generate_random_names(names_count):
    names = []
    for i in range(names_count):
        names.append(str(uuid.uuid4()))
    return names


def from_cvat_to_yolo(data_folder: str, output_images_folder: str, output_labels_folder: str, label_names: list):
    xml_path = os.path.join(data_folder, "annotations.xml")
    frames_count = get_frames_count_from_xml(xml_path)
    input_images_path = os.path.join(data_folder, "images") 
    
    first_image_path = None
    image_names = []
    idx = 0
    for filename in os.listdir(input_images_path):
        file_path = os.path.join(input_images_path, filename)
        if os.path.isfile(file_path):
            if idx <= 0:
                first_image_path = file_path
                image_names.append(filename)
            idx += 1
    
    if first_image_path is not None:
        img = cv2.imread("first_image_path")
        height, width, _ = img.shape
        shape = (width, height)
    else:
        shape = (1920, 1080)
    
    random_names = generate_random_names(frames_count)

    from_xml_to_yolo(xml_path, output_labels_folder, label_names, shape, random_names=random_names)

    for image_name in image_names:
        index = re.findall("[/d]+", image_name)
        new_name = random_names[index]
        os.rename(image_name, new_name)

    move_files(input_images_path, output_images_folder)

    

# def from_lane_line_batch_to_yolo(lane_line_batch: list, ouput_path: str):
#     output_string = ""
#     for lane_line in lane_line_batch:
        


# def from_xml_to_yolo(xml_path: str, output_path: str):
#     lane_lines = get_lane_lines_from_xml(xml_path)

    # lane_lines
