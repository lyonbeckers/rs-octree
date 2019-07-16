struct Point {
    x: f32,
    y: f32,
    z: f32
}

impl std::default::Default for Point {
    fn default() -> Self {
        Point {
            x: std::f32::MAX,
            y: std::f32::MAX,
            z: std::f32::MAX
        }
    }
}

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl PartialEq for Point {
    fn eq(&self, rhs : &Point) -> bool {
        self.x == rhs.x && self.y == rhs.y && self.z == rhs.z
    }
}

impl std::ops::Add for Point {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z
        }
    }
}

impl std::ops::Sub for Point {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z
        }
    }
}

impl std::ops::Div<f32> for Point {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs
        }
    }
}

impl Copy for Point {

}

impl Clone for Point {
    fn clone(&self) -> Point {
        *self
    }
}

impl Point {

    fn new(x :f32, y :f32, z :f32) -> Point {
        Point {
            x, y, z
        }
    }
}

struct Voxel {
    point: Point
}

impl Default for Voxel {
    fn default() -> Voxel {
        Voxel {
            point: Point::default()
        }
    }
}

impl Copy for Voxel {

}

impl Clone for Voxel {
    fn clone(&self) -> Voxel {
        *self
    }
}

struct AABB {
    center: Point,
    half_dimension: f32,
    min: Point,
    max: Point
}

impl Copy for AABB {

}

impl Clone for AABB {
    fn clone(&self) -> AABB {
        *self
    }
}

impl AABB {
    fn new(center: Point, half_dimension: f32) -> AABB {
        AABB {
            center, 
            half_dimension,
            max: Point::new(center.x + half_dimension, 
                center.y + half_dimension,
                center.z + half_dimension),
            min: Point::new(center.x - half_dimension, 
                center.y - half_dimension,
                center.z - half_dimension)
        }
    }

    fn contains_point(&self, point: Point) -> bool {
        point.x >= self.min.x && point.x <= self.max.x
        && point.y >= self.min.y && point.y <= self.max.y
        && point.z >= self.min.z && point.z <= self.max.z
    }

    fn intersects_bounds(&self, aabb: AABB) -> bool {
        self.min.x <= aabb.max.x && self.max.x >= aabb.min.x
        && self.min.y <= aabb.max.y && self.max.y >= aabb.min.y
        && self.min.z <= aabb.max.z && self.max.z >= aabb.min.z
    }
}

struct Octree{
    aabb: AABB,
    num_voxels: u32,
    voxels: [Voxel; 8],
    children: Vec<Octree>,
}

impl Octree {
    const MAX_SIZE: u32 = 8;

    fn new(aabb: AABB) -> Octree {
        Octree {
            aabb,
            num_voxels: 0,
            voxels: [Voxel::default(); 8],
            children: vec![],
        }
    }

    fn subdivide(&mut self) {

        println!("Subdividing {} ", self.aabb.center);

        let downbackleft = AABB::new(
            self.aabb.center + (self.aabb.min - self.aabb.center)/2.0
            , self.aabb.half_dimension/2.0);

        let downbackright = AABB::new(
            self.aabb.center + (Point::new(self.aabb.max.x, self.aabb.min.y, self.aabb.min.z) - self.aabb.center)/2.0
            , self.aabb.half_dimension/2.0);

        let downforwardleft = AABB::new(
            self.aabb.center + (Point::new(self.aabb.min.x, self.aabb.min.y, self.aabb.max.z) - self.aabb.center)/2.0
            , self.aabb.half_dimension/2.0);

        let downforwardright = AABB::new(
            self.aabb.center + (Point::new(self.aabb.max.x, self.aabb.min.y, self.aabb.max.z) - self.aabb.center)/2.0
            , self.aabb.half_dimension/2.0);

        let upbackleft = AABB::new(
            self.aabb.center + (Point::new(self.aabb.min.x, self.aabb.max.y, self.aabb.min.z) - self.aabb.center)/2.0
            , self.aabb.half_dimension/2.0);

        let upbackright = AABB::new(
            self.aabb.center + (Point::new(self.aabb.max.x, self.aabb.max.y, self.aabb.min.z) - self.aabb.center)/2.0
            , self.aabb.half_dimension/2.0);

        let upforwardleft = AABB::new(
            self.aabb.center + (Point::new(self.aabb.min.x, self.aabb.max.y, self.aabb.max.z) - self.aabb.center)/2.0
            , self.aabb.half_dimension/2.0);

        let upforwardright = AABB::new(
            self.aabb.center + (self.aabb.max - self.aabb.center)/2.0
            , self.aabb.half_dimension/2.0);

        self.children = vec![
            Octree::new(downbackleft),
            Octree::new(downbackright),
            Octree::new(downforwardleft),
            Octree::new(downforwardright),
            Octree::new(upbackleft),
            Octree::new(upbackright),
            Octree::new(upforwardleft),
            Octree::new(upforwardright)
        ];

    }

    fn insert(&mut self, voxel: Voxel) -> bool{  

        if !self.aabb.contains_point(voxel.point) {
            return false
        }

        if self.num_voxels < Octree::MAX_SIZE && self.children.len() == 0 {
            self.voxels[self.num_voxels as usize] = voxel;
            println!("Inserted {} as voxel number {} in {}, {}", self.voxels[self.num_voxels as usize].point, self.num_voxels, self.aabb.center, self.aabb.half_dimension);

            self.num_voxels = self.num_voxels + 1;
            return true
        }

        if self.children.len() == 0 {
            
            self.subdivide();
        }

        for child in &mut self.children {
            if child.insert(voxel) {
                return true
            }
        }

        false
        
    }

    fn query_range(&mut self, range: AABB) -> Vec<Voxel> {

        let mut voxels_in_range = vec![];
        
        if !self.aabb.intersects_bounds(range) {
            return voxels_in_range
        }

        if self.num_voxels == 0 {
            return voxels_in_range
        }

        for &voxel in self.voxels.iter() {
            if voxel.point == Point::default() {
                continue
            }
            if self.aabb.contains_point(voxel.point) {
                voxels_in_range.push(voxel);
            }
        }

        if self.children.len() == 0 {
            return voxels_in_range
        }

        for child in &mut self.children {
            voxels_in_range.append(&mut child.query_range(range))
        }

        voxels_in_range
    }
}

fn main() {
    let aabb = AABB::new(Point::new(5.0,5.0,5.0), 5.0);
    let mut node = Octree::new(aabb);
    
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..5 {
                node.insert(Voxel {
                    point: Point::new(x as f32,y as f32,z as f32)
                });
            }
        }
    }

    println!("{}", &node.query_range(aabb).len());
}
