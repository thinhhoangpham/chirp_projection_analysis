from chirp_python.bin2d import Bin2D
from chirp_python.rect import Rect
from chirp_python.hdr import HDR

class RectangularCover:
    def __init__(self, bin2d: Bin2D, min_covered_points, purity_threshold):
        self.bin2d = bin2d
        self.min_covered_points = min_covered_points
        self.binner = bin2d.binner
        self.purity_threshold = purity_threshold
        self.bins = self.binner.get_bins()
        self.n_bins = self.binner.get_num_bins()
        self.class_index = bin2d.chdr.class_index
        self.i1 = 0
        self.i2 = 0
        self.j1 = 0
        self.j2 = 0
        self.best_odds = 0
        self.class_count = 0
        self.other_count = 0

    def compute(self):
        total_covered = 0
        for _ in range(10):  # number of rectangles allowed
            best_cover = None
            best_count = 0
            for i in range(self.n_bins):
                for j in range(self.n_bins):
                    if not self.bins[i][j].is_covered and self.bins[i][j].is_pure(self.class_index, self.purity_threshold):
                        self._cover(i, j)
                        if self.class_count > best_count:
                            best_cover = self._make_rectangle()
                            best_count = self.class_count
            
            if best_count < self.min_covered_points:
                break
            else:
                self.bin2d.chdr.add(HDR(best_cover))
                self._tag_covered_bins(best_cover)
                total_covered += best_count
        return total_covered

    def _make_rectangle(self):
        x = float(self.i1) / self.n_bins
        y = float(self.j1) / self.n_bins
        w = float(self.i2 - self.i1 + 1) / self.n_bins
        h = float(self.j2 - self.j1 + 1) / self.n_bins
        return Rect(x, y, w, h)

    def _tag_covered_bins(self, rect: Rect):
        i1 = int(rect.x * self.n_bins)
        j1 = int(rect.y * self.n_bins)
        i2 = i1 + int(rect.width * self.n_bins)
        j2 = j1 + int(rect.height * self.n_bins)
        for i in range(i1, i2):
            for j in range(j1, j2):
                self.bins[i][j].is_covered = True

    def _cover(self, ix, iy):
        self.i1 = ix
        self.i2 = self.i1
        self.j1 = iy
        self.j2 = self.j1
        self.class_count = self.bins[ix][iy].class_counts[self.class_index]
        self.other_count = self.bins[ix][iy].count - self.class_count
        self.best_odds = float(self.class_count + 1) / (self.other_count + 1)
        
        border_count = 0
        while True:
            border_count = 0
            border_count += self._look_up()
            border_count += self._look_right()
            border_count += self._look_down()
            border_count += self._look_left()
            if border_count == 0:
                break
        
        if self.i1 == self.i2 and self.j1 == self.j2:
            self.class_count = 0

    def _look_up(self):
        if self.j2 > self.n_bins - 2:
            return 0
        c_count = self.class_count
        o_count = self.other_count
        for i in range(self.i1, self.i2 + 1):
            b = self.bins[i][self.j2 + 1]
            c_count += b.class_counts[self.class_index]
            o_count += b.count - b.class_counts[self.class_index]
            if c_count <= b.count or o_count > 10:
                return 0
        
        odds = float(c_count + 1) / (o_count + 1)
        if odds < self.best_odds:
            return 0
        
        self.best_odds = odds
        self.class_count = c_count
        self.other_count = o_count
        self.j2 += 1
        return 1

    def _look_down(self):
        if self.j1 < 1:
            return 0
        c_count = self.class_count
        o_count = self.other_count
        for i in range(self.i1, self.i2 + 1):
            b = self.bins[i][self.j1 - 1]
            c_count += b.class_counts[self.class_index]
            o_count += b.count - b.class_counts[self.class_index]
            if c_count <= b.count or o_count > 10:
                return 0
        
        odds = float(c_count + 1) / (o_count + 1)
        if odds < self.best_odds:
            return 0
            
        self.best_odds = odds
        self.class_count = c_count
        self.other_count = o_count
        self.j1 -= 1
        return 1

    def _look_left(self):
        if self.i1 < 1:
            return 0
        c_count = self.class_count
        o_count = self.other_count
        for j in range(self.j1, self.j2 + 1):
            b = self.bins[self.i1 - 1][j]
            c_count += b.class_counts[self.class_index]
            o_count += b.count - b.class_counts[self.class_index]
            if c_count <= b.count or o_count > 10:
                return 0
        
        odds = float(c_count + 1) / (o_count + 1)
        if odds < self.best_odds:
            return 0
            
        self.best_odds = odds
        self.class_count = c_count
        self.other_count = o_count
        self.i1 -= 1
        return 1

    def _look_right(self):
        if self.i2 > self.n_bins - 2:
            return 0
        c_count = self.class_count
        o_count = self.other_count
        for j in range(self.j1, self.j2 + 1):
            b = self.bins[self.i2 + 1][j]
            c_count += b.class_counts[self.class_index]
            o_count += b.count - b.class_counts[self.class_index]
            if c_count <= b.count or o_count > 10:
                return 0
        
        odds = float(c_count + 1) / (o_count + 1)
        if odds < self.best_odds:
            return 0
            
        self.best_odds = odds
        self.class_count = c_count
        self.other_count = o_count
        self.i2 += 1
        return 1
