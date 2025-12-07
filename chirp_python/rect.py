class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getCenterX(self):
        return self.getX() + self.getWidth() / 2.0

    def getCenterY(self):
        return self.getY() + self.getHeight() / 2.0

    def getMinX(self):
        return self.getX()

    def getMinY(self):
        return self.getY()

    def getMaxX(self):
        return self.getX() + self.getWidth()

    def getMaxY(self):
        return self.getY() + self.getHeight()

    def contains(self, x, y):
        x0 = self.getX()
        y0 = self.getY()
        return (x >= x0 and y >= y0 and x < x0 + self.getWidth() and y < y0 + self.getHeight())
