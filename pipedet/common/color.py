


COLOR_NAME_TO_RGB = {'BLACK': (0, 0, 0),
              'WHITE': (255,255,255),
              'RED': (255,0,0),
              'LIME': (0,255,0),
              'BLUE': (0,0,255),
              'YELLOW': (255,255,0),
              'CYAN': (0,255,255),
              'MAGENTA': (255,0,255),
              'CORAL': (255,127,80),
              'ORANGE': (255,165,0),
              'GREEN': (0,128,0),
              'AQUA_MARINE': (127,255,212),
              'DEEP_SKY_BLUE': (0,191,255),
              'VIOLET': (238,130,238),
              'DEEP_PINK': (255,20,147)}


#HINO above dic is correct if (,,) = (R,G,B), so
COLOR_NAME_TO_BGR = {}
for c in COLOR:
    rgb = COLOR_NAME_TO_RGB[c]
    COLOR_NAME_TO_BGR[c] = (rgb[2],rgb[1],rgb[0])
