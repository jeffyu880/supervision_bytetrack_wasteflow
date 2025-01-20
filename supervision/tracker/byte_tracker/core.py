from typing import List, Tuple

import numpy as np
import importlib

from supervision.detection.core import Detections
from supervision.detection.utils import box_iou_batch
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter
# importlib.reload(supervision.tracker.byte_tracker)
from supervision.tracker.byte_tracker.nonlinear_kalman_filter import AdvancedNonLinearKalmanFilter
from supervision.tracker.byte_tracker.kalman_filter_simplified import KalmanFilterS
from supervision.tracker.byte_tracker.single_object_track import STrack, TrackState
from supervision.tracker.byte_tracker.utils import IdCounter
# importlib.reload(AdvancedNonLinearKalmanFilter)
from scipy.optimize import linear_sum_assignment



def interpolate_bboxes(bbox_prev, bbox_next, steps=1):
    """
    Interpolates bounding boxes between previous and current frame.

    Args:
        bbox_prev (np.ndarray): Bounding box of the previous frame in TLBR format.
        bbox_next (np.ndarray): Bounding box of the current frame in TLBR format.
        steps (int): Number of intermediate steps (frames) to interpolate.

    Returns:
        List[np.ndarray]: List of interpolated bounding boxes.
    """
    if steps > 1:
        return [
            bbox_prev + (bbox_next - bbox_prev) * (i / steps) for i in range(1, steps)
        ]
    else:
        return [bbox_next]  # If steps == 1, no interpolation, return the next bbox directly
    
def detections2boxes(detections: Detections) -> np.ndarray:
    """
    Convert Supervision Detections to numpy tensors for further computation.
    Args:
        detections (Detections): Detections/Targets in the format of sv.Detections.
    Returns:
        (np.ndarray): Detections as numpy tensors as in
            `(x_min, y_min, x_max, y_max, confidence, class_id)` order.
    """
    return np.hstack(
        (
            detections.xyxy,
            detections.confidence[:, np.newaxis],
            detections.class_id[:, np.newaxis],
        )
    )

def compute_iou(bbox1, bbox2):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = bbox1
    x1_t, y1_t, x2_t, y2_t = bbox2

    # Calculate area of both bounding boxes
    area_bbox1 = (x2 - x1) * (y2 - y1)
    area_bbox2 = (x2_t - x1_t) * (y2_t - y1_t)

    # Calculate the coordinates of the intersection box
    x1_int = max(x1, x1_t)
    y1_int = max(y1, y1_t)
    x2_int = min(x2, x2_t)
    y2_int = min(y2, y2_t)

    # Calculate area of intersection
    intersection_area = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)

    # Calculate IoU
    iou = intersection_area / float(area_bbox1 + area_bbox2 - intersection_area)
    return iou


def calculate_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return np.array([center_x, center_y])

def compute_distance(center1, center2):
    return np.linalg.norm(center1 - center2)

def compute_cost_matrix(previous_bboxes, current_bboxes, image_width, image_height):
    cost_matrix = np.full((len(previous_bboxes), len(current_bboxes)), float(1e6))
    
    image_diagonal = np.sqrt(image_width**2 + image_height**2)  # Calculate the diagonal of the image
    
    for i, bbox_prev in enumerate(previous_bboxes):
        center_prev = calculate_center(bbox_prev)
        
        for j, bbox_curr in enumerate(current_bboxes):
            center_curr = calculate_center(bbox_curr)
            
            # Compute IoU
            # iou = [0,1] : 1 is perfect matching
            iou = compute_iou(bbox_prev, bbox_curr)
            iou_cost = 1 - iou  # Cost based on IoU
            
            # Compute Distance
            distance = compute_distance(center_prev, center_curr)
            distance_normalized = distance / image_diagonal  # Normalize the distance

            # normalized distance in range [0,1] : 1 is furthest away
            # In a 480 x 480 image, 30fps, a paper moved 38 pixels in one frame
            # max a paper can move across 1 frame is lets say 48 pixels, so normalize distance is 0.1

            if iou >= 0.1 and distance_normalized <= 0.1:
                # Weight factors for IoU and Distance
                alpha = 0.5  # IoU weight
                beta = 0.5  # Distance weight
                
                # Compute final cost combining IoU and distance
                final_cost = alpha * iou_cost + beta * distance_normalized
                cost_matrix[i, j] = final_cost
            
            ####### CHECK TO SEE IF THIS IS A GOOD METRIC ########
              # Apply IoU threshold
            # if iou < iou_threshold:
            #     cost_matrix[i, j] = float('inf')  # Reject if IoU is below threshold
            # # Apply Distance threshold
            # elif dist > distance_threshold:
            #     cost_matrix[i, j] = float('inf')  # Reject if distance is too large
            # else:
            #     cost_matrix[i, j] = 1 - iou  # Use IoU for cost calculation


    return cost_matrix

def assign_bboxes(previous_bboxes, current_bboxes, current_scores, current_class_ids, cost_matrix):
    # Use the Hungarian algorithm to find the optimal assignment (minimizing cost)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    ########### TODO ###########
    ## Figure out a threshold to not match if the distances or IoU are too large

    # Create the zipped list of matched bounding boxes
    matches = [(previous_bboxes[i], current_bboxes[j], current_scores[j], current_class_ids[j]) for i, j in zip(row_indices, col_indices)]
    return matches

class ByteTrack:
    """
    Initialize the ByteTrack object.

    <video controls>
        <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/annotate-video-with-traces.mp4" type="video/mp4">
    </video>

    Parameters:
        track_activation_threshold (float): Detection confidence threshold
            for track activation. Increasing track_activation_threshold improves accuracy
            and stability but might miss true detections. Decreasing it increases
            completeness but risks introducing noise and instability.
        lost_track_buffer (int): Number of frames to buffer when a track is lost.
            Increasing lost_track_buffer enhances occlusion handling, significantly
            reducing the likelihood of track fragmentation or disappearance caused
            by brief detection gaps.
        minimum_matching_threshold (float): Threshold for matching tracks with detections.
            Increasing minimum_matching_threshold improves accuracy but risks fragmentation.
            Decreasing it improves completeness but risks false positives and drift.
        frame_rate (int): The frame rate of the video.
        minimum_consecutive_frames (int): Number of consecutive frames that an object must
            be tracked before it is considered a 'valid' track.
            Increasing minimum_consecutive_frames prevents the creation of accidental tracks from
            false detection or double detection, but risks missing shorter tracks.
    """  # noqa: E501 // docs

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
    ):
        self.track_activation_threshold = track_activation_threshold
        self.minimum_matching_threshold = minimum_matching_threshold

        self.frame_id = 0
        self.det_thresh = self.track_activation_threshold + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.kalman_filter = KalmanFilter()
        self.shared_kalman = KalmanFilter()

        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []

        # Warning, possible bug: If you also set internal_id to start at 1,
        # all traces will be connected across objects.
        self.internal_id_counter = IdCounter()
        self.external_id_counter = IdCounter(start_id=1)

        self.total_tracked_tracks = 0
        self.fps = 10
        self.previous_bboxes = None
        self.image_width = 480
        self.image_height = 480
    
    def update_with_detections(self, detections: Detections) -> Detections:
        """
        Updates the tracker with the provided detections and returns the updated
        detection results.

        Args:
            detections (Detections): The detections to pass through the tracker.

        Example:
            ```python
            import supervision as sv
            from ultralytics import YOLO

            model = YOLO(<MODEL_PATH>)
            tracker = sv.ByteTrack()

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            def callback(frame: np.ndarray, index: int) -> np.ndarray:
                results = model(frame)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = tracker.update_with_detections(detections)

                labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

                annotated_frame = box_annotator.annotate(
                    scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels)
                return annotated_frame

            sv.process_video(
                source_path=<SOURCE_VIDEO_PATH>,
                target_path=<TARGET_VIDEO_PATH>,
                callback=callback
            )
            ```
        """
        tensors = np.hstack(
            (
                detections.xyxy,
                detections.confidence[:, np.newaxis],
            )
        )

        ###########################################################################################

        # custom_tensors = detections2boxes(detections=detections)

        # current_bboxes = np.array([det[:4] for det in custom_tensors])  # Current frame's bounding boxes
        # print("current_bboxes: ", current_bboxes)
        # # calculate area of bboxes
        # # Calculate area for each bounding box
        # # areas = (current_bboxes[:, 2] - current_bboxes[:, 0]) * (current_bboxes[:, 3] - current_bboxes[:, 1])

        # # # Store areas with the current_bboxes as extra column
        # # bboxes_with_area = np.hstack([current_bboxes, areas.reshape(-1, 1)])

        # current_scores = np.array([det[4] for det in custom_tensors])   # Current frame's confidence scores
        # current_class_ids = np.array([det[5] for det in custom_tensors]) # Current frame's class IDs
        # print("current class ID: ", current_class_ids)
        # tracks = []

        # # If this is not the first frame and the FPS is lower than 20, interpolate
        # if self.previous_bboxes is not None:
        #     if len(detections.xyxy)==0:
        #         print("no detections but has previous boxes")
        #         tracks = self.update_with_tensors(tensors=tensors)
        #     # Check the FPS to determine whether to perform interpolation
        #     if self.fps == 20:
        #         steps = 5 ##3  # Perform two interpolations (between previous and current frame)
        #         ##if steps = 2, then only one interpolation is done
        #     else:
        #         steps = 1  # No interpolation if FPS is 20 or higher
            
        #     # Create a cost matrix based on IoU and distance
        #     # assign pairs of current and previous bboxes using the hungarian matching algorithm for closeness in distance and size
        #     cost_matrix =  compute_cost_matrix(self.previous_bboxes, current_bboxes, self.image_width, self.image_height)
        #     print(cost_matrix)
        #     # if no matches: cost_matrix is inf:
        #     if np.all(cost_matrix == float('inf')):
        #         print("No valid matches, reinitializing tracking.")
        #         # Handle by resetting tracks or choosing fallback behavior
        #         # tracks = []  # or do some other fallback action
        #         tracks = self.update_with_tensors(tensors=tensors)
        #     else:
        #         # List to hold all fake detections
        #         all_fake_detections = []

        #         matched_bboxes = assign_bboxes(self.previous_bboxes, current_bboxes, current_scores, current_class_ids, cost_matrix) 
        #         for prev_bbox, curr_bbox, score, class_id in matched_bboxes:
        #             interpolated_bboxes = interpolate_bboxes(prev_bbox, curr_bbox, steps) 
        #             print("Interpolated Boxes: ", interpolated_bboxes)
        #             print("bbox prev: ", prev_bbox)
        #             print("bbox next: ", curr_bbox)
                    
        #             for bbox in interpolated_bboxes:
        #                 print("bbbox: ", bbox)
        #                 # Here, use the Kalman filter to predict and update with the interpolated positions
        #                 # print(("score: ", score))
        #                 fake_detection = np.hstack([bbox, 1.0, class_id])  # Create a fake detection with a confidence of 1.0
        #                 fake_detections = np.expand_dims(fake_detection, axis=0)
        #                 print(fake_detections)
        #                 all_fake_detections.append(fake_detection)

        #         all_fake_detections_array = np.array(all_fake_detections)
        #         # run update on fake detections once
        #         tracks = self.update_with_tensors(all_fake_detections_array)
        
        # # elif len(detections.xyxy)==0:
        # #     print("no detections")
        # #     tracks = []
        
        # else: 
        #     print("no previous boxes")
        #     tracks = self.update_with_tensors(tensors=tensors)
        # self.previous_bboxes = current_bboxes
    
        # #################################################################################################
        
        tracks = self.update_with_tensors(tensors=tensors)
        # print("grabbing")
        
        if len(tracks) > 0:
            detection_bounding_boxes = np.asarray([det[:4] for det in tensors])
            track_bounding_boxes = np.asarray([track.tlbr for track in tracks])

            ious = box_iou_batch(detection_bounding_boxes, track_bounding_boxes)

            iou_costs = 1 - ious

            matches, _, _ = matching.linear_assignment(iou_costs, 0.5)
            detections.tracker_id = np.full(len(detections), -1, dtype=int)
            for i_detection, i_track in matches:
                detections.tracker_id[i_detection] = int(
                    tracks[i_track].external_track_id
                )

            return detections[detections.tracker_id != -1]

        else:
            detections = Detections.empty()
            detections.tracker_id = np.array([], dtype=int)

            return detections

    def reset(self) -> None:
        """
        Resets the internal state of the ByteTrack tracker.

        This method clears the tracking data, including tracked, lost,
        and removed tracks, as well as resetting the frame counter. It's
        particularly useful when processing multiple videos sequentially,
        ensuring the tracker starts with a clean state for each new video.
        """
        self.frame_id = 0
        self.internal_id_counter.reset()
        self.external_id_counter.reset()
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []

    def update_with_tensors(self, tensors: np.ndarray) -> List[STrack]:
        """
        Updates the tracker with the provided tensors and returns the updated tracks.

        Parameters:
            tensors: The new tensors to update with.

        Returns:
            List[STrack]: Updated tracks.
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = tensors[:, 4]
        bboxes = tensors[:, :4]

        remain_inds = scores > self.track_activation_threshold
        inds_low = scores > 0.1
        inds_high = scores < self.track_activation_threshold

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        # print("detected elements: ", len(scores))
        # print("detected elements kept: ", len(scores_keep) + len(scores_second))

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    score_keep,
                    self.minimum_consecutive_frames,
                    self.shared_kalman,
                    self.internal_id_counter,
                    self.external_id_counter,
                )
                for (tlbr, score_keep) in zip(dets, scores_keep)
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]

        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # print("unconfirmed: ", unconfirmed)
        # print("tracked tracks: ", tracked_stracks)

        self.tracked_tracks += tracked_stracks

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_tracks(tracked_stracks, self.lost_tracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.shared_kalman)
        dists = matching.iou_distance(strack_pool, detections)

        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.minimum_matching_threshold
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    score_second,
                    self.minimum_consecutive_frames,
                    self.shared_kalman,
                    self.internal_id_counter,
                    self.external_id_counter,
                )
                for (tlbr, score_second) in zip(dets_second, scores_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.state = TrackState.Lost
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)

        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = TrackState.Removed
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = TrackState.Removed
                removed_stracks.append(track)

        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.Tracked
        ]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_starcks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks = removed_stracks
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        output_stracks = [track for track in self.tracked_tracks if track.is_activated]

        return output_stracks


def joint_tracks(
    track_list_a: List[STrack], track_list_b: List[STrack]
) -> List[STrack]:
    """
    Joins two lists of tracks, ensuring that the resulting list does not
    contain tracks with duplicate internal_track_id values.

    Parameters:
        track_list_a: First list of tracks (with internal_track_id attribute).
        track_list_b: Second list of tracks (with internal_track_id attribute).

    Returns:
        Combined list of tracks from track_list_a and track_list_b
            without duplicate internal_track_id values.
    """
    seen_track_ids = set()
    result = []

    for track in track_list_a + track_list_b:
        if track.internal_track_id not in seen_track_ids:
            seen_track_ids.add(track.internal_track_id)
            result.append(track)

    return result


def sub_tracks(track_list_a: List[STrack], track_list_b: List[STrack]) -> List[int]:
    """
    Returns a list of tracks from track_list_a after removing any tracks
    that share the same internal_track_id with tracks in track_list_b.

    Parameters:
        track_list_a: List of tracks (with internal_track_id attribute).
        track_list_b: List of tracks (with internal_track_id attribute) to
            be subtracted from track_list_a.
    Returns:
        List of remaining tracks from track_list_a after subtraction.
    """
    tracks = {track.internal_track_id: track for track in track_list_a}
    track_ids_b = {track.internal_track_id for track in track_list_b}

    for track_id in track_ids_b:
        tracks.pop(track_id, None)

    return list(tracks.values())


def remove_duplicate_tracks(
    tracks_a: List[STrack], tracks_b: List[STrack]
) -> Tuple[List[STrack], List[STrack]]:
    pairwise_distance = matching.iou_distance(tracks_a, tracks_b)
    matching_pairs = np.where(pairwise_distance < 0.15)

    duplicates_a, duplicates_b = set(), set()
    for track_index_a, track_index_b in zip(*matching_pairs):
        time_a = tracks_a[track_index_a].frame_id - tracks_a[track_index_a].start_frame
        time_b = tracks_b[track_index_b].frame_id - tracks_b[track_index_b].start_frame
        if time_a > time_b:
            duplicates_b.add(track_index_b)
        else:
            duplicates_a.add(track_index_a)

    result_a = [
        track for index, track in enumerate(tracks_a) if index not in duplicates_a
    ]
    result_b = [
        track for index, track in enumerate(tracks_b) if index not in duplicates_b
    ]

    return result_a, result_b
