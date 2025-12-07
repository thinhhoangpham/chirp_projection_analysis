from chirp_python.rect import Rect

class HDR:
    def __init__(self, rect: Rect):
        self.rect = rect

    def contains_point(self, x, y):
        return self.rect.contains(x, y)

    def centroid_infinity_distance(self, x, y):
        return max(abs(x - self.rect.getCenterX()),
                   abs(y - self.rect.getCenterY()))

    def rect_infinity_distance(self, x, y):
        xmin = 0
        if x < self.rect.getX():
            xmin = self.rect.getX() - x
        elif x > self.rect.getMaxX():
            xmin = x - self.rect.getMaxX()
        
        ymin = 0
        if y < self.rect.getY():
            ymin = self.rect.getY() - y
        elif y > self.rect.getMaxY():
            ymin = y - self.rect.getMaxY()

        return max(xmin, ymin)
