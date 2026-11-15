from __future__ import annotations

from typing import Annotated

from pydantic import Field

NSD_BUCKET = "natural-scenes-dataset"
NSD_REGION = "us-east-2"

# Completed sessions per subject. The metadata CSV lists the *planned* 40 for everyone.
NSD_SESSIONS: dict[int, int] = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}
SUBJECTS: tuple[int, ...] = tuple(NSD_SESSIONS)

N_NSD_IMAGES = 73_000
TRIALS_PER_SESSION = 750
BETA_SCALE = 300.0  # betas are stored as int16 = percent signal change * 300
STIM_SIZE = 425     # NSD stimulus resolution (px)

SubjectId = Annotated[int, Field(ge=1, le=8)]
