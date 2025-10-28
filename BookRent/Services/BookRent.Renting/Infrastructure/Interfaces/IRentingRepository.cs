using BookRent.Renting.Models;

namespace BookRent.Renting.Infrastructure.Interfaces;

public interface IRentingRepository
{
    Task<bool> ReturnBookAsync(Guid bookId, Guid customerId);
    Task<bool> ReturnBookAsync(Guid orderId);
    Task<Guid> RentBookAsync(RentedBook order);
    Task<List<RentedBook>> GetAllRentsAsync();
    Task<List<RentedBook>> GetOverDueOrdersAsync(Guid customerId);
    Task<List<RentedBook>> GetOpenRentsAsync(Guid customerId);
    Task<List<RentedBook>> GetRentHistoryAsync(Guid customerId);

    Task<bool> EditBookCounterAsync(Guid bookId, int newMaxCount);



}