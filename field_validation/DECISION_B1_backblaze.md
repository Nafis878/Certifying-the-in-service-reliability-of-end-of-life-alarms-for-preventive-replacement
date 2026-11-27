# Decision on dataset B1 (Backblaze Drive Stats): rejected under the pre-specified admissibility rule

Date: 2026-11-27. Status: decided before any Backblaze data file was downloaded, opened or analysed.

## The pre-specified rule

Protocol v1 (`protocol.md`, hashed before any field data were obtained), section A, B1:

> Known threat: Backblaze's failure flag may include drives pulled on SMART evidence (failure partly alarm-defined) -
> check against Backblaze's published definition before analysis; if it cannot be defended, B1 is rejected.

## What Backblaze publishes about its failure flag

1. The Q3 2025 Drive Stats report says that a Drive Stats failure is a drive that leaves the reporting population, or
   whose serial number is the primary serial of a drive-replacement work ticket. Its main replacement indicator is a drive
   that responds to reads with uncorrectable medium errors; a monitoring tool (drive sentinel) flags read/write errors
   above a threshold and, on SATA Storage Pods, removes access to the drive. The same report describes drives pulled for a
   firmware update being counted as failed although the failure was not mechanical.
   https://www.backblaze.com/blog/backblaze-drive-stats-for-q3-2025/
2. Backblaze's SMART-stats article says that SMART 187 is one of the attributes it uses to decide that a drive has failed,
   and that once it rises above zero the drive is scheduled for replacement.
   https://www.backblaze.com/blog/hard-drive-smart-stats/
3. Backblaze conference material lists, among its failure criteria, SMART statistics above its thresholds, and
   distinguishes reactive failures from proactive failures triggered by SMART statistics.
   https://snia.org/sites/default/files/2025-05/SNIA-SDC21-Klein-Facts-Figures-and-Insights-from-250000-Hard-Drives.pdf
4. Nothing in the published data separates these removal types; the dataset has a single `failure` flag.

## Decision

The end-of-life event in B1 is partly defined by the operator's own alarm on the error counts that the candidate policies
use as inputs (SMART 5, 187, 188, 197, 198), and it also includes non-mechanical removals. For this study an
alarm-triggered preventive removal would be counted as an in-service failure, so F(l), the accepted-yet-failed share
and the certificates would not measure what the manuscript defines. The event cannot be separated in the published
data, so the definition cannot be defended. **B1 is rejected under the pre-specified rule.** No Backblaze file was
downloaded or analysed; no Backblaze result exists.

## Consequence for the article

The fleet evidence rests on one dataset (SCANIA Component X). The manuscript says so in its Limitations and Future Work,
and Supplementary Section S22 records this decision. A second fleet dataset needs an event that is not defined by an
operator alarm on the policy's inputs. One candidate is the Alibaba SSD failure data (Han et al., FAST 2021), whose
failures come from trouble tickets for whole-drive or partial-drive failure. Its definition, content and size must be
checked, and a new protocol must be written and hashed, before any of its data are seen.
