# ID PROCESSING BY BUKAYO
# v0.01 - 30/08/24
# v1.00 - 31/08/24 
# v1.01 - 01/09/24
# v1.02 - 10/09/24
# v2.00 - 13/09/24
# v2.01 - 14/09/24
# v2.02 - 17/09/24
# v2.03 - 18/09/24
# Note: Warnings are muted (Always double check output for error)
# Not Yet Implemented: Validation and Error Handling

import os
import re
import cv2
import time
import numpy
import pandas
import colorama
from PIL import Image, ImageDraw, ImageFont

skipped_images = []
skipped_lrn = []
no_contact_person = []

def text_center_position(draw, text, font, left_bound, right_bound):
    bounding_box = draw.textbbox((0, 0), text, font = font)
    text_width = bounding_box[2] - bounding_box[0]
    text_horizontal_position = left_bound + (right_bound - left_bound - text_width) / 2
    return text_horizontal_position

def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    bar = '#' * int(percent) + '·' * (100 - int(percent))
    print(f'{colorama.Fore.YELLOW}\r[{bar}] {percent:.2f}%', end = '\r')
    
    if progress == total:
        print(f'{colorama.Fore.GREEN}\r[{bar}] {percent:.2f}%') 
    
def crop_face(image_path):
    print(colorama.Fore.CYAN + '[1/6]: FACE DETECTION AND CROPPING')
    
    image_list = os.listdir(image_path)
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in image_list if f.lower().endswith(image_extensions)]
    if len(image_files) < 1:
        print(colorama.Fore.RED + 'Error: No image files found.')
        exit()

    output_path = f'{image_path}/output/cropped_images'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    face_detection_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_crop_padding = 750

    counter = 0
    progress_bar(counter, len(image_files))
    for image in image_files:
        student_image_path = os.path.join(image_path, image)

        loaded_student_image = cv2.imread(student_image_path)
        gray_image = cv2.cvtColor(loaded_student_image, cv2.COLOR_BGR2GRAY)

        faces = face_detection_model.detectMultiScale(gray_image, scaleFactor = 1.3, minNeighbors = 6, minSize=(600, 600))

        for _, (x, y, w, h) in enumerate(faces):
            
            x_new = max(0, x - face_crop_padding)
            y_new = max(0, y - face_crop_padding)
            w_new = min(loaded_student_image.shape[1] - x_new, w + 2 * face_crop_padding)
            h_new = min(loaded_student_image.shape[0] - y_new, h + 2 * face_crop_padding)
            cropped_face = loaded_student_image[y_new:y_new+h_new, x_new:x_new+w_new]

            output_image_path = os.path.join(output_path, image)
            cv2.imwrite(output_image_path, cropped_face)
            
        counter += 1
        progress_bar(counter, len(image_files))

def circular_face(image_path):
    print(colorama.Fore.CYAN + '[2/6]: TRIMMING CROPPED FACES TO CIRCULAR IMAGES')

    cropped_faces_path = os.path.join(image_path, 'output/cropped_images')
    image_list = os.listdir(cropped_faces_path)

    output_path = f'{image_path}/output/front_layout'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    counter = 0
    progress_bar(counter, len(image_list))
    for cropped_face in image_list:
        cropped_face_path = os.path.join(cropped_faces_path, cropped_face)

        image = cv2.imread(cropped_face_path, cv2.IMREAD_UNCHANGED)

        height, width = image.shape[:2]

        mask = numpy.zeros((height, width), dtype = numpy.uint8)
        center = (width // 2, height // 2)
        radius = min(center[0], center[1])
        cv2.circle(mask, center, radius, 255, -1)

        circular_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        circular_image[:, :, 3] = mask

        filename = f'{cropped_face.split('.')[0]}.PNG'
        output_image_path = os.path.join(output_path, filename)
        cv2.imwrite(output_image_path, circular_image)

        counter += 1
        progress_bar(counter, len(image_list))

def overlay_face(image_path, front_layout_path):
    print(colorama.Fore.CYAN + '[3/6]: OVERLAYING FACES TO FRONT LAYOUT')

    save_path = os.path.join(image_path, 'output/front_layout')
    image_list = os.listdir(save_path)

    counter = 0
    progress_bar(counter, len(image_list))
    for circular_face in image_list:
        cropped_face_path = os.path.join(save_path, circular_face)

        image = cv2.imread(cropped_face_path, cv2.IMREAD_UNCHANGED)
        front_layout = cv2.imread(front_layout_path)

        image = cv2.resize(image, (2235, 2235))
        overlay_height, overlay_width = image.shape[:2]

        horizontal_position = 1070
        vertical_position = 1450

        overlay_image = image[:, :, :3]
        overlay_mask = image[:, :, 3:]
        background_mask = 255 - overlay_mask

        overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
        
        background_region = front_layout[vertical_position:vertical_position+overlay_height, horizontal_position:horizontal_position+overlay_width]

        blended_part = (background_region * (background_mask / 255)) + (overlay_image * (overlay_mask / 255))

        front_layout[vertical_position:vertical_position+overlay_height, horizontal_position:horizontal_position+overlay_width] = blended_part
        
        output_image_path = os.path.join(save_path, circular_face)
        cv2.imwrite(output_image_path, front_layout)

        counter += 1
        progress_bar(counter, len(image_list))

def overlay_name(image_path, data_path):
    print(colorama.Fore.CYAN + '[4/6]: OVERLAYING STUDENT NAME TO FRONT LAYOUT')

    save_path = os.path.join(image_path, 'output/front_layout')
    layout_list = os.listdir(save_path)
    
    dataframe = pandas.read_excel(data_path, skiprows = 6, usecols = 'E:H', dtype = {'Student ID': str})
    dataframe = dataframe.dropna(how = 'all')

    counter = 0
    progress_bar(counter, len(layout_list))
    for front_layout in layout_list:  
        row = dataframe.loc[dataframe['Student ID'] == front_layout.split('.')[0]]
        if row.empty:
            skipped_images.append(front_layout)
            counter += 1
            progress_bar(counter, len(layout_list))
            continue
        row = row.to_dict()
        row_num = list(row['Student ID'].keys())[0]
        
        middle_name = f'{row['Middle Name'][row_num][0]}.' if pandas.notna(row['Middle Name'][row_num]) else ''
        name = f'{row['First Name'][row_num]} {middle_name} {row['Last Name'][row_num]}'
        name = name.upper()

        image_overlay = Image.new('RGBA', (6000, 400), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image_overlay)
        font = ImageFont.truetype('font/Montserrat/Montserrat-SemiBold.ttf', 220)
        draw.text((0, 0), name, font = font, fill = (255, 255, 255, 255))
        bounding_box = image_overlay.getbbox()

        if bounding_box:
            trimmed_image = image_overlay.crop(bounding_box)
            trimmed_width, trimmed_height = trimmed_image.size

            max_width = 3140
            if trimmed_width > max_width:
                new_width = max_width
                new_height = trimmed_height
                condensed_text = trimmed_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                condensed_text = trimmed_image
        else:
            condensed_text = trimmed_image
            
        front_image_layout_path = os.path.join(save_path, front_layout)
        image_layout = Image.open(front_image_layout_path).convert('RGBA')

        overlay_width, _ = condensed_text.size

        left_bound = 635
        right_bound = 3775
        center_horizontal = (left_bound + right_bound) // 2
        x = center_horizontal - (overlay_width // 2)
        y = 4070

        combined_image = image_layout.copy()
        combined_image.paste(condensed_text, (x, y), condensed_text)

        output_image_path = os.path.join(save_path, front_layout)
        combined_image.save(output_image_path)
        
        counter += 1
        progress_bar(counter, len(layout_list))

def overlay_front_info(image_path, data_path):
    print(colorama.Fore.CYAN + '[5/6]: OVERLAYING STUDENT INFORMATION IN FRONT LAYOUT')

    save_path = os.path.join(image_path, 'output/front_layout')
    layout_list = os.listdir(save_path)
    
    dataframe = pandas.read_excel(
        data_path, 
        skiprows = 6, 
        usecols = 'C, E, O:P', 
        dtype = {'LRN': str, 'Student ID': str})
    dataframe = dataframe.dropna(how = 'all')
    dataframe['LRN'] = dataframe['LRN'].str.replace('LRN-', '', regex = False)

    counter = 0
    progress_bar(counter, len(layout_list))
    for front_layout in layout_list:
        row = dataframe.loc[dataframe['Student ID'] == front_layout.split('.')[0]]
        if row.empty:
            counter += 1
            progress_bar(counter, len(layout_list))
            continue
        row = row.to_dict()
        row_num = list(row['LRN'].keys())[0]
        
        image_path = os.path.join(save_path, front_layout)
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        designation_font = 'font/Montserrat/Montserrat-MediumItalic.ttf'
        lrn_id_font = 'font/ChaparralPro/ChaparralPro-Bold.otf'

        left_bound = 635
        right_bound = 3775

        des_font = ImageFont.truetype(designation_font, 180)
        grade = f'{row['Grade Level / Degree Level'][row_num]}'
        track = f' - {row['Track & Strand / Course'][row_num]}'
        if row['Track & Strand / Course'][row_num] == 'ACADEMIC TRACKS - Accounting, Business Management':
            track = ' - ABM'
        elif row['Track & Strand / Course'][row_num] == 'ACADEMIC TRACKS - Humanities and Social Sciences':
            track = ' - HUMSS'
        elif row['Track & Strand / Course'][row_num] == 'ACADEMIC TRACKS - Science, Technology, Engineering, and Mathematics':
            track = ' - STEM'
        elif row['Track & Strand / Course'][row_num] == 'TECHNICAL VOCATIONAL LIVELIHOOD TRACKS - Home Economics':
            track = ' - TVL'
        elif row['Track & Strand / Course'][row_num] == '-':
            track = ''
        designation = f'{grade}{track}'
        
        lrn_font = ImageFont.truetype(lrn_id_font, 225)
        if pandas.notna(row['LRN'][row_num]):
            lrn_number = f'LRN: {row['LRN'][row_num]}' 
        else:
            lrn_number = 'LRN:'
            skipped_lrn.append(front_layout.split('.')[0])        
        
        id_no_font = ImageFont.truetype(lrn_id_font, 175)
        student_number = f'Student No.: {row['Student ID'][row_num]}'
        
        text_horizontal_position = text_center_position(draw, designation, des_font, left_bound, right_bound)
        text_vertical_position = 4300
        draw.text((text_horizontal_position, text_vertical_position), designation, font = des_font, fill = 'white')
        
        text_horizontal_position = text_center_position(draw, lrn_number, lrn_font, left_bound, right_bound)
        text_vertical_position = 4890
        draw.text((text_horizontal_position, text_vertical_position), lrn_number, font = lrn_font, fill = 'white')
        
        text_horizontal_position = text_center_position(draw, student_number, id_no_font, left_bound, right_bound)
        text_vertical_position = 5150
        draw.text((text_horizontal_position, text_vertical_position), student_number, font = id_no_font, fill = 'white')

        output_image_path = os.path.join(save_path, front_layout)
        image.save(output_image_path)

        counter += 1
        progress_bar(counter, len(layout_list))

def overlay_back_info(image_path, data_path, back_layout_path):
    print(colorama.Fore.CYAN + '[6/6]: OVERLAYING PARENT/GUARDIAN INFORMATION TO THE LAYOUT')

    save_path = f'{image_path}/output/back_layout'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    dataframe = pandas.read_excel(
        data_path, skiprows = 6, 
        usecols = 'E, V:W, Y:Z, AB:AC, AE:AF', 
        dtype = {'Person to notify Contact Number': str, 
                 'Mother\'s Contact Number': str, 
                 'Father\'s Contact Number': str,
                 'Guardian\'s Contact Number': str})
    dataframe = dataframe.dropna(how = 'all')

    font_path_name = 'font/Montserrat/Montserrat-Bold.ttf'
    font_path_phone = 'font/Montserrat/Montserrat-SemiBold.ttf'
    
    name_size = 200
    name_position = 400
    name_font = ImageFont.truetype(font_path_name, name_size)

    phone_size = 200
    phone_position = 630
    phone_font = ImageFont.truetype(font_path_phone, phone_size)

    counter = 0
    progress_bar(counter, len(dataframe))
    for _, row in dataframe.iterrows():
        image = Image.open(back_layout_path)
        draw = ImageDraw.Draw(image)
        width, _ = image.size
        
        student_number = row['Student ID']
        
        if not (pandas.isna(row['Person to notify']) and pandas.isna(row['Person to notify Contact Number'])):
            name = f'{row['Person to notify']}'
            phone = f'{row['Person to notify Contact Number']}'
        elif not (pandas.isna(row['Mother']) and pandas.isna(row['Mother\'s Contact Number'])):
            name = f'{row['Mother']}'
            phone = f'{row['Mother\'s Contact Number']}'
        elif not (pandas.isna(row['Father']) and pandas.isna(row['Father\'s Contact Number'])):
            name = f'{row['Father']}'
            phone = f'{row['Father\'s Contact Number']}'
        elif not (pandas.isna(row['Guardian']) and pandas.isna(row['Guardian\'s Contact Number'])):
            name = f'{row['Guardian']}'
            phone = f'{row['Guardian\'s Contact Number']}'
        else:
            name = ''
            phone = ''
            no_contact_person.append(student_number)

        if not (name == '' and phone == ''):
            name = name.upper()
            match = re.match(r'^(.*?)\,\s(.*?)(?:\s(\w+\.)\s*)?$', name)
            firstname = match.group(2)
            middlename = match.group(3)
            lastname = match.group(1)
            
            extension = ''
            index = firstname.find('JR')
            if index != -1:
                firstname = firstname[:index - 1]
                extension = f' JR'
            
            index = firstname.find('.')
            if index != -1:
                if firstname[index + 1] != ' ':
                    firstname = firstname[:index + 1] + ' ' + firstname[index + 1:]
            name = f'{firstname} {middlename} {lastname}{extension}'
            
        if len(phone) == 11:
            phone = phone[:4] + ' ' + phone[4:]
            phone = phone[:8] + ' ' + phone[8:]
        
        bounding_box = draw.textbbox((0, 0), name, font = name_font)
        text_width = bounding_box[2] - bounding_box[0]
        text_horizontal_position = (width - text_width) / 2
        draw.text((text_horizontal_position, name_position), name, font = name_font, fill = 'black')

        bounding_box = draw.textbbox((0, 0), phone, font = phone_font)
        text_width = bounding_box[2] - bounding_box[0]
        text_horizontal_position = (width - text_width) / 2
        draw.text((text_horizontal_position, phone_position), phone, font = phone_font, fill = 'black')

        output_image_path = os.path.join(save_path, f'{student_number}.JPG')
        image.save(output_image_path)

        counter += 1
        progress_bar(counter, len(dataframe))
        
def main(image_path, data_path, front_layout_path, back_layout_path):
    os.system(f'title Automated ID Layout Tool by BUKAYO :) v2.03')
    os.system('cls')
    
    colorama.init()
    print(colorama.Fore.YELLOW + "This project was lazily made by BUKAYO :) [18/9/24]")

    start_time = time.time()

    dataframe = pandas.read_excel(data_path, skiprows=6, usecols='O:P')
    dataframe = dataframe.dropna()
    if dataframe['Track & Strand / Course'].iloc[0] != '-':
        print(f'{colorama.Fore.WHITE}\nGRADE LEVEL / DEGREE LEVEL: {dataframe['Grade Level / Degree Level'].iloc[0]} - {dataframe['Track & Strand / Course'].iloc[0]}')
    else:
        print(f'{colorama.Fore.WHITE}\nGRADE LEVEL / DEGREE LEVEL: {dataframe['Grade Level / Degree Level'].iloc[0]}')

    # crop_face(image_path)
    # circular_face(image_path)
    # overlay_face(image_path, front_layout_path)
    # overlay_name(image_path, data_path)
    # overlay_front_info(image_path, data_path)
    overlay_back_info(image_path, data_path, back_layout_path)

    print(f'{colorama.Fore.GREEN}\nID DATA AND IMAGE PROCESSING COMPLETE')
    
    if (len(skipped_lrn)) > 0:
        print(f'{colorama.Fore.RED}\nWARNING: {len(skipped_lrn)} student/s has no LRN from the datasheet.')
        for student_id in skipped_lrn:
            print(f'{colorama.Fore.WHITE}- {student_id}')

    if len(skipped_images) > 0:
        print(f'{colorama.Fore.RED}\nWARNING: {len(skipped_images)} image/s has no matching Student ID from the datasheet.')
        for filename in skipped_images:
            print(f'{colorama.Fore.WHITE}- {filename}')
    
    if len(no_contact_person) > 0:
        print(f'{colorama.Fore.RED}\nWARNING: {len(no_contact_person)} student/s has no contact information from datasheet.')
        for student_id in no_contact_person:
            print(f'{colorama.Fore.WHITE}- {student_id}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time / 60)
    seconds = int(elapsed_time - (minutes * 60))
    if seconds < 10:
        seconds = f'0{seconds}'
    print(f'{colorama.Fore.YELLOW}\nELAPSED TIME: {minutes}:{seconds}')
    
    colorama.deinit()

if __name__ == '__main__':
    main(
        image_path = 'images/Grade 3 St. Augustine',
        data_path = 'xlsx/GRADE 3 - ST. AUGUSTINE.xlsx', 
        front_layout_path = 'layout/IDTemplate(2)SeniorHigh.png',
        back_layout_path = 'layout/IDTemplateBack.png',
    )