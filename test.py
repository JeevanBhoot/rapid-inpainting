# from PIL import Image
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torch.utils.data import DataLoader, random_split
# from model import *
# from loss import *
# from dataset import *

# img_transforms_lst = [
#     transforms.Resize((512, 512)),
#     transforms.ToTensor()
# ]
# mask_transforms_lst = [
#     transforms.ToTensor()
# ]
# img_transform = transforms.Compose(img_transforms_lst)
# mask_transform = transforms.Compose(mask_transforms_lst)

# img = Image.load('0.png')
# img_t = img_transform(img)
# mask = generate_random_mask(512, 512)
# mask_t = mask_transform(mask)

# x = img_t * mask_t

#print(get_image_files('/data/cornucopia/jsb212/seg-dataset/eval_inpaint'))

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

def getIntersection1(headA, headB):
    if headA and headB:
        pointerA, pointerB = headA, headB
        while pointerA is not pointerB:
            pointerA = headB if pointerA is None else pointerA.next
            pointerB = headA if pointerB is None else pointerB.next
            print(pointerA, pointerB)
        return pointerA
    else:
        return None
    
def getIntersection2(headA, headB):
    if headA and headB:
        setA = set()
        pointerA = headA
        while pointerA:
            setA.add(pointerA)
            pointerA = pointerA.next

        pointerB = headB
        while pointerB:
            if pointerB in setA:
                return pointerB
            pointerB = pointerB.next
        return None
    else:
        return None
    
llistA = LinkedList()
llistA.head = ListNode(1)
second = ListNode(2)
third = ListNode(3)
llistA.head.next = second  # Link first node with second
second.next = third 


llistB = LinkedList()
llistB.head = ListNode(4)
secondB = ListNode(4)
thirdB = ListNode(4)
fourthB = ListNode(4)
llistB.head.next = secondB  # Link first node with second
second.next = thirdB
third.next = fourthB

print(getIntersection1(llistA.head, llistB.head))