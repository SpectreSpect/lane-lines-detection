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
        label = int(label_names_dict_inversed[str(track.attrib['label'])])
        # label = str(track.attrib['label'])

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


def get_boxes_from_image_cvat(path: str):
    tree = ET.parse(path)
    root = tree.getroot()

    image_elements = root.findall('.//image')
    bouding_box_batches = []
    for image_element in image_elements:
        name = str(image_element.attrib['name'])
        width = int(image_element.attrib['width'])
        height = int(image_element.attrib['height'])

        box_elements = image_element.findall(".//box")
        bounding_boxes = []
        for box_element in box_elements:
            label = str(box_element.attrib['label'])
            xtl = float(box_element.attrib['xtl'])
            ytl = float(box_element.attrib['ytl'])
            xbr = float(box_element.attrib['xbr'])
            ybr = float(box_element.attrib['ybr'])

            xtl /= width
            ytl /= height
            xbr /= width
            ybr /= height

            x = (xbr + xtl) / 2.0
            y = (ytl + ybr) / 2.0

            box_width = xbr - xtl
            box_height = ybr - ytl

            bouding_box = BoundingBox(label, x, y, box_width, box_height, name)
            bounding_boxes.append(bouding_box) 
        bouding_box_batches.append(bounding_boxes)
    return bouding_box_batches


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

        output_string += str(label) + " "
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


def get_video_shape(video_path: str) -> list:
    video = cv2.VideoCapture(video_path)

    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video.release()

    return [width, height]


def from_cvat_to_yolo(output_images_folder: str, output_labels_folder: str, label_names: list, 
                      input_video_path: str = None, 
                      xml_path: str = None, 
                      input_labels_path: str = None,
                      input_videos_path: str = None, verbose=1):
    if (input_labels_path is not None) and (input_videos_path is not None):
        # label_file_names = get_file_names(input_labels_path)
        video_file_names = get_file_names(input_videos_path)

        for idx, video_file_name in enumerate(video_file_names):
            if verbose == 1:
                print(f"{idx}) {video_file_name}")
            video_file_name_without_ext, _ = os.path.splitext(video_file_name)
            label_file_name = str(video_file_name_without_ext) + ".xml"


            video_path = os.path.join(input_videos_path, video_file_name)
            label_file_path = os.path.join(input_labels_path, label_file_name)

            frames_count = get_frames_count_from_xml(label_file_path)
            random_names = generate_random_names(frames_count)
            shape = get_video_shape(video_path)

            save_video_into_frames(video_path, output_images_folder, random_names)
            from_xml_to_yolo(label_file_path, output_labels_folder, label_names, shape, random_names=random_names)
    else:
        frames_count = get_frames_count_from_xml(xml_path)
        random_names = generate_random_names(frames_count)
        shape = get_video_shape(input_video_path)

        save_video_into_frames(input_video_path, output_images_folder, random_names)
        from_xml_to_yolo(xml_path, output_labels_folder, label_names, shape, random_names=random_names)
    

def add_boxes_from_cvat_image(labels_path: str, cvat_xml_path: str):
    pass