"""
tools/ffmpeg/ffmpeg_core.py 单元测试
"""


class TestComputeTotalDuration:
    """compute_total_duration() 视频总时长计算"""

    def test_no_transition(self):
        from tools.ffmpeg.ffmpeg_core import compute_total_duration
        total = compute_total_duration(
            durations=[10.0, 20.0, 30.0],
            xfade_type=None,
            transition_duration=1.0,
            n=3,
        )
        assert total == 60.0

    def test_with_xfade(self):
        """xfade 会各损失 transition_duration 一次"""
        from tools.ffmpeg.ffmpeg_core import compute_total_duration
        total = compute_total_duration(
            durations=[10.0, 10.0, 10.0],
            xfade_type="fade",
            transition_duration=1.0,
            n=3,
        )
        # 30 - 2*1 = 28 (3 个视频, 2 个过渡点)
        assert total == 28.0

    def test_single_video_no_transition(self):
        """单个视频无 xfade = 原时长"""
        from tools.ffmpeg.ffmpeg_core import compute_total_duration
        total = compute_total_duration(
            durations=[15.0],
            xfade_type="fade",
            transition_duration=1.0,
            n=1,
        )
        assert total == 15.0

    def test_two_videos_with_fade(self):
        from tools.ffmpeg.ffmpeg_core import compute_total_duration
        total = compute_total_duration(
            durations=[10.0, 10.0],
            xfade_type="fade",
            transition_duration=2.0,
            n=2,
        )
        # 20 - 1*2 = 18
        assert total == 18.0
