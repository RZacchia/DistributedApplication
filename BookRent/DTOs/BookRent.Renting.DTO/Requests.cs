namespace BookRent.Renting.DTO;

public record ReturnBookRequest(Guid UserId, Guid BookId);
public record RentBookRequest(Guid UserId, Guid BookId);
public record EditBookCounterRequest(Guid BookId, int Counter);