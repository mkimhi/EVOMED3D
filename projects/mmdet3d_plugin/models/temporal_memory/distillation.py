import torch.nn.functional as F


class DistillationLoss:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, student, teacher):
        return self.weight * self._smooth_l1(student, self._detach(teacher))

    def _smooth_l1(self, student, teacher):
        if isinstance(student, (list, tuple)):
            losses = [
                F.smooth_l1_loss(student_item, teacher_item)
                for student_item, teacher_item in zip(student, teacher)
            ]
            return sum(losses) / len(losses)
        return F.smooth_l1_loss(student, teacher)

    def _detach(self, teacher):
        if isinstance(teacher, list):
            return [item.detach() for item in teacher]
        if isinstance(teacher, tuple):
            return tuple(item.detach() for item in teacher)
        return teacher.detach()
