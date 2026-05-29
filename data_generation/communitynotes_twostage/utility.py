"""
Utility functions for filtering ratings before matrix factorization training.
Extracted from process_data.py for custom MF implementations.
"""

from typing import Tuple
import numpy as np
import pandas as pd

import constants as c #, note_status_history


def remove_duplicate_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
  """Drop duplicate ratings, then assert that there is exactly one rating per noteId per raterId.

  Args:
      ratings (pd.DataFrame) with possible duplicated ratings

  Returns:
      pd.DataFrame: ratings, with one record per userId, noteId.
  """
  # Construct a new DataFrame to avoid SettingWithCopyWarning
  ratings = pd.DataFrame(ratings.drop_duplicates())

  numRatings = len(ratings)
  numUniqueRaterIdNoteIdPairs = len(ratings.groupby([c.raterParticipantIdKey, c.noteIdKey]).head(1))
  assert (
    numRatings == numUniqueRaterIdNoteIdPairs
  ), f"Only {numUniqueRaterIdNoteIdPairs} unique raterId,noteId pairs but {numRatings} ratings"
  return ratings


def remove_duplicate_notes(notes: pd.DataFrame) -> pd.DataFrame:
  """Remove duplicate notes, then assert that there is only one copy of each noteId.

  Args:
      notes (pd.DataFrame): with possible duplicate notes

  Returns:
      notes (pd.DataFrame) with one record per noteId
  """
  # Construct a new DataFrame to avoid SettingWithCopyWarning
  notes = pd.DataFrame(notes.drop_duplicates())

  numNotes = len(notes)
  numUniqueNotes = len(np.unique(notes[c.noteIdKey]))
  assert (
    numNotes == numUniqueNotes
  ), f"Found only {numUniqueNotes} unique noteIds out of {numNotes} notes"

  return notes


# def _filter_misleading_notes(
#   notes: pd.DataFrame,
#   ratings: pd.DataFrame,
#   noteStatusHistory: pd.DataFrame,
#   logging: bool = True,
# ) -> pd.DataFrame:
#   """
#   This function actually filters ratings (not notes), based on which notes they rate.

#   Filter out ratings of notes that say the Tweet isn't misleading.
#   Also filter out ratings of deleted notes, unless they were deleted after
#     c.deletedNotesTombstoneLaunchTime, and appear in noteStatusHistory.

#   Args:
#       notes (pd.DataFrame): _description_
#       ratings (pd.DataFrame): _description_
#       noteStatusHistory (pd.DataFrame): _description_
#       logging (bool, optional): _description_. Defaults to True.

#   Returns:
#       pd.DataFrame: filtered ratings
#   """
#   ratings = ratings.merge(
#     noteStatusHistory[[c.noteIdKey, c.createdAtMillisKey, c.classificationKey]],
#     on=c.noteIdKey,
#     how="left",
#     suffixes=("", "_nsh"),
#   )

#   deletedNoteKey = "deletedNote"
#   notDeletedMisleadingKey = "notDeletedMisleading"
#   deletedButInNSHKey = "deletedButInNSH"
#   createdAtMillisNSHKey = c.createdAtMillisKey + "_nsh"

#   ratings[deletedNoteKey] = pd.isna(ratings[c.classificationKey])
#   ratings[notDeletedMisleadingKey] = np.invert(ratings[deletedNoteKey]) & (
#     ratings[c.classificationKey] == c.notesSaysTweetIsMisleadingKey
#   )
#   ratings[deletedButInNSHKey] = ratings[deletedNoteKey] & np.invert(
#     pd.isna(ratings[createdAtMillisNSHKey])
#   )

#   deletedNotInNSH = (ratings[deletedNoteKey]) & pd.isna(ratings[createdAtMillisNSHKey])
#   notDeletedNotMisleadingOldUI = (
#     ratings[c.classificationKey] == c.noteSaysTweetIsNotMisleadingKey
#   ) & (ratings[createdAtMillisNSHKey] <= c.notMisleadingUILaunchTime)
#   notDeletedNotMisleadingNewUI = (
#     ratings[c.classificationKey] == c.noteSaysTweetIsNotMisleadingKey
#   ) & (ratings[createdAtMillisNSHKey] > c.notMisleadingUILaunchTime)

#   if logging:
#     print(
#       f"Preprocess Data: Filter misleading notes, starting with {len(ratings)} ratings on {len(np.unique(ratings[c.noteIdKey]))} notes"
#     )
#     print(
#       f"  Keeping {ratings[notDeletedMisleadingKey].sum()} ratings on {len(np.unique(ratings.loc[ratings[notDeletedMisleadingKey],c.noteIdKey]))} misleading notes"
#     )
#     print(
#       f"  Keeping {ratings[deletedButInNSHKey].sum()} ratings on {len(np.unique(ratings.loc[ratings[deletedButInNSHKey],c.noteIdKey]))} deleted notes that were previously scored (in note status history)"
#     )
#     print(
#       f"  Removing {notDeletedNotMisleadingOldUI.sum()} ratings on {len(np.unique(ratings.loc[notDeletedNotMisleadingOldUI, c.noteIdKey]))} older notes that aren't deleted, but are not-misleading."
#     )
#     print(
#       f"  Removing {deletedNotInNSH.sum()} ratings on {len(np.unique(ratings.loc[deletedNotInNSH, c.noteIdKey]))} notes that were deleted and not in note status history (e.g. old)."
#     )

#   ratings = ratings[
#     ratings[notDeletedMisleadingKey] | ratings[deletedButInNSHKey] | notDeletedNotMisleadingNewUI
#   ]
#   ratings = ratings.drop(
#     columns=[
#       createdAtMillisNSHKey,
#       c.classificationKey,
#       deletedNoteKey,
#       notDeletedMisleadingKey,
#       deletedButInNSHKey,
#     ]
#   )
#   return ratings


def filter_ratings(ratings: pd.DataFrame, logging: bool = True) -> pd.DataFrame:
  """Apply min number of ratings for raters & notes. Instead of iterating these filters
  until convergence, simply stop after going back and force once.

  This is the critical filter that produces the final training set for matrix factorization.

  Args:
      ratings (pd.DataFrame): unfiltered ratings
      logging (bool, optional): debug output. Defaults to True.

  Returns:
      pd.DataFrame: filtered ratings ready for MF training
  """

  n = ratings.groupby(c.noteIdKey).size().reset_index()
  notesWithMinNumRatings = n[n[0] >= c.minNumRatersPerNote]

  ratingsNoteFiltered = ratings.merge(notesWithMinNumRatings[[c.noteIdKey]], on=c.noteIdKey)

  if logging:
    print("Filter notes and ratings with too few ratings")
    print(
      "  After Filtering Notes w/less than %d Ratings, Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d"
      % (
        c.minNumRatersPerNote,
        len(ratingsNoteFiltered),
        len(np.unique(ratingsNoteFiltered[c.noteIdKey])),
        len(np.unique(ratingsNoteFiltered[c.raterParticipantIdKey])),
      )
    )
  r = ratingsNoteFiltered.groupby(c.raterParticipantIdKey).size().reset_index()
  ratersWithMinNumRatings = r[r[0] >= c.minNumRatingsPerRater]

  ratingsDoubleFiltered = ratingsNoteFiltered.merge(
    ratersWithMinNumRatings[[c.raterParticipantIdKey]], on=c.raterParticipantIdKey
  )
  if logging:
    print(
      "  After Filtering Raters w/less than %s Notes, Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d"
      % (
        c.minNumRatingsPerRater,
        len(ratingsDoubleFiltered),
        len(np.unique(ratingsDoubleFiltered[c.noteIdKey])),
        len(np.unique(ratingsDoubleFiltered[c.raterParticipantIdKey])),
      )
    )
  n = ratingsDoubleFiltered.groupby(c.noteIdKey).size().reset_index()
  notesWithMinNumRatings = n[n[0] >= c.minNumRatersPerNote]
  ratingsForTraining = ratingsDoubleFiltered.merge(
    notesWithMinNumRatings[[c.noteIdKey]], on=c.noteIdKey
  )
  if logging:
    print(
      "  After Final Filtering of Notes w/less than %d Ratings, Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d"
      % (
        c.minNumRatersPerNote,
        len(ratingsForTraining),
        len(np.unique(ratingsForTraining[c.noteIdKey])),
        len(np.unique(ratingsForTraining[c.raterParticipantIdKey])),
      )
    )
  return ratingsForTraining
