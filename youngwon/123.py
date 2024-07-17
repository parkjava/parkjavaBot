import cv2
import numpy as np

def detect(img):
    img = cv2.imread('/Users/youngwonchoi/Desktop/20240307/01.FinalProject/ML/image')
    height, width = img.shape
    channel = 1
    
    # Morphology Operation
    StructuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    topHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, StructuringElement)
    blackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, StructuringElement)
    img_topHat = cv2.add(img, topHat)
    img = cv2.subtract(img_topHat, blackHat)
    
    # Gaussian Blurring
    blur = cv2.GaussianBlur(img, (5, 5), 2)
    
    # Adaptive Thresholding
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
    91, 3)

    # Find and Drawing Contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_result = np.zeros((height, width, channel), dtype=np.uint8)
    cv2.drawContours(contour_result, contours, -1, (255, 255, 255))
    contour_result = np.zeros((height, width, channel), dtype=np.uint8)

    # Drawing Rectangle with Contours
    cntDict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(contour_result, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cntDict.append({'contour': contour, 'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + (w / 2), 'cy': y + (h /
        2)})
        
        # Plate Size Assumption
    MINAREA = 100       # Minimum Area
    MINWIDTH = 4        # Minimum Width
    MINHEIGHT = 2       # Minimum Height
    MINRATIO = 0.25     # Minimum Ratio (Width / Height)
    MAXRATIO = 0.9      # Maximum Ratio (Width / Height)
        
    # Candidates of the Plate
    cntPossible = []
    cnt = 0
    for item in cntDict:
        # Calculate Conditions
        area = item['w'] * item['h'] # 가로 * 세로 넓이
        ratio = item['w'] / item['h'] # 가로 / 세로 비율
        
        # Check Conditions
        if area > MINAREA and item['w'] > MINWIDTH and item['h'] > MINHEIGHT and MINRATIO < ratio < MAXRATIO:
            item['index'] = cnt
            cnt += 1
            cntPossible.append(item)
            
    contour_result = np.zeros((height, width, channel), dtype=np.uint8)

    # Draw Rectangle on Candidates
    for cnt in cntPossible:
        cv2.rectangle(contour_result, (cnt['x'], cnt['y']), (cnt['x'] + cnt['w'], cnt['y'] + cnt['h']), (255, 255,
    255), 2)
            # Numeral Arrange Appearance Conditions
    MAXDIAG = 5             # Average Distance from Center
    MAXANGLE = 12           # Maximum Angle between Contours
    MAXAREADIFF = 0.5       # Maximum Area Difference between Contours
    MAXWIDTHDIFF = 0.8      # Maximum Width Difference between Contours
    MAXHEIGHTDIFF = 0.2     # Maximum Height Difference between Contours
    MINCHARACTER = 5        # Minimum Counts of Numbers

    # Recursive Function to Find Characters
    def FindCharacter(cntList):
        result = []

        for item1 in cntList:
            match = []
            for item2 in cntList:
                # Same Index Contours are not be compared
                if item1['index'] == item2['index']:
                    continue

                # Distance from Center Point
                dx = abs(item1['cx'] - item2['cx'])
                dy = abs(item1['cy'] - item2['cy'])

                # First Contour's Diagonal Length
                diagLength = np.sqrt(item1['w'] ** 2 + item1['h'] ** 2)

                # Distance between Vectors
                distance = np.linalg.norm(np.array([item1['cx'], item1['cy']]) - np.array([item2['cx'], item2['cy']]))

                # Angle Calculations
                if dx == 0:
                    angle = 90
                else:
                    angle = np.degrees(np.arctan(dy / dx))
                    
                DiffArea = abs(item1['w'] * item1['h'] - item2['w'] * item2['h']) / (item1['w'] * item1['h'])
                DiffWidth = abs(item1['w'] - item2['w']) / item1['w']
                DiffHeight = abs(item1['h'] - item2['h']) / item1['h']

                # Check Conditions
                if distance < diagLength * MAXDIAG and angle < MAXANGLE and DiffArea < MAXAREADIFF and DiffWidth < MAXWIDTHDIFF and DiffHeight < MAXHEIGHTDIFF:
                    match.append(item2['index'])
                
            match.append(item1['index'])

            if len(match) < MINCHARACTER:
                continue

            result.append(match)

            # Unmatched Contours
            unmatch = []
            for item3 in cntList:
                if item3['index'] not in match:
                    unmatch.append(item3['index'])

            # Only Take Same Contours
            unmatch = np.take(cntPossible, unmatch)

            # Function Called Recursively
            recursion = FindCharacter(unmatch)

            for i in recursion:
                result.append(i)

            break

        return result

    # Index of Character Contours
    indexResult = FindCharacter(cntPossible)

    matchResult = []
    for indexList in indexResult:
        matchResult.append(np.take(cntPossible, indexList))

    # Visualize Possible Contours
    possibleResult = np.zeros((height, width, channel), dtype=np.uint8)
        
    for items in matchResult:
        for item in items:
            cv2.rectangle(possibleResult, (item['x'], item['y']), (item['x'] + item['w'], item['y'] + item['h']), (255, 255, 255), 2)
        
    # Get Number Plate Informations
    for i, items in enumerate(matchResult):
        # Sortion to Axis X
        sortion = sorted(items, key=lambda x: x['cx'])

        return sortion