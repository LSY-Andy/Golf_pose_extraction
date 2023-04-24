from util.frame_data import Frames


class BaseDetector():
    '''
    An extractor to read from a source and get all the pose estimations
    '''
    def __init__(self, source: str = '', display=False) -> None:
        # TODO make it as an argumentation
        self.display = display
        self.set_source(source)

    def set_source(self, source: str) -> None:
        # TODO input data type checking
        self.source = source

    def video_extract(self) -> Frames:
        raise NotImplementedError

    def image_extract(self) -> Frames:
        raise NotImplementedError

    def extract(self):
        # TODO: the return type for both video_extract and image_extract are currently list(list(np.ndarray)) which is quite stupid. Create some new self defined datatypes in util and use them instead
        ext = self.source.split('.')[-1]
        if ext == 'mp4':
            result = self.video_extract()
        else:
            result = self.image_extract()
        return result
