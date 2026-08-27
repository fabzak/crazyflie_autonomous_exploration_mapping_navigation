import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import yaml

class MapSaver(Node):
    def __init__(self):
        super().__init__('custom_map_saver')
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        self.saved = False

    def map_callback(self, msg):
        if self.saved:
            return

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = msg.info.origin

        data = list(msg.data)

        pgm_path = 'moja_mapa.pgm'
        yaml_path = 'moja_mapa.yaml'

        with open(pgm_path, 'wb') as f:
            f.write(bytearray(f'P5\n{width} {height}\n255\n', 'ascii'))

            for y in range(height):
                for x in range(width):
                    i = x + (height - y - 1) * width
                    val = data[i]

                    if val == -1:
                        pixel = 205
                    elif val >= 65:
                        pixel = 0
                    else:
                        pixel = 254

                    f.write(bytearray([pixel]))

        map_metadata = {
            'image': pgm_path,
            'mode': 'trinary',
            'resolution': float(resolution),
            'origin': [
                float(origin.position.x),
                float(origin.position.y),
                float(origin.position.z)
            ],
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.25
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(map_metadata, f, default_flow_style=False)

        self.get_logger().info(f'Mapa spremljena: {pgm_path}, {yaml_path}')
        self.saved = True
        rclpy.shutdown()

def main():
    rclpy.init()
    node = MapSaver()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
